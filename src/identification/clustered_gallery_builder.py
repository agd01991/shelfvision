from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .crop_extractor import CropRecord, extract_crops_from_predictions_file
from .feature_extractor import VisualFeatureExtractor
from .gallery_manager import build_sku_gallery


@dataclass
class ProvisionalSkuItem:
    candidate_id: str
    source_image: str
    source_object_id: int
    source_crop_path: str
    candidate_image_path: str
    score: float
    width: int
    height: int
    source_type: str
    has_mask: bool


@dataclass
class ClusteredDemoGalleryItem:
    sku_id: str
    sku_name: str
    candidate_id: str
    source_image: str
    source_object_id: int
    source_crop_path: str
    gallery_image_path: str
    score: float
    width: int
    height: int
    source_type: str
    ref_index: int
    cluster_size: int
    cluster_mean_similarity: float
    cluster_min_similarity: float
    is_primary_ref: bool


@dataclass
class ClusteredGallerySummary:
    predictions_json: str
    images_dir: str
    gallery_dir: str
    gallery_csv: str
    crops_dir: str
    provisional_dir: str
    requested_sku_count: int
    created_sku_count: int
    extracted_crops_count: int
    selected_crops_count: int
    provisional_sku_count: int
    gallery_refs_count: int
    duplicate_refs_count: int
    skipped_duplicate_crops_count: int
    dropped_candidates_count: int
    min_score: float
    min_width: int
    min_height: int
    use_masks: bool
    deduplicate: bool
    dedup_threshold: float
    max_refs_per_sku: int
    gallery_build_mode: str
    cluster_merge_threshold: float
    cluster_strong_merge_threshold: float
    cluster_min_similarity: float
    cluster_pair_report_threshold: float
    cluster_max_candidates: int
    status: str
    warning: str = ""


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra
        return ra


def _crop_size(path: str | Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        return 0, 0
    height, width = image.shape[:2]
    return int(width), int(height)


def _clear_old_demo_skus(gallery_dir: Path, prefix: str) -> None:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for child in gallery_dir.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            shutil.rmtree(child)


def _select_candidates(
    crops: List[CropRecord],
    min_score: float,
    min_width: int,
    min_height: int,
    max_candidates: int,
) -> List[tuple[CropRecord, int, int]]:
    candidates: List[tuple[CropRecord, int, int]] = []
    for crop in crops:
        width, height = _crop_size(crop.crop_path)
        if width < min_width or height < min_height:
            continue
        if crop.score < min_score:
            continue
        candidates.append((crop, width, height))

    candidates.sort(key=lambda item: (item[0].score, item[1] * item[2]), reverse=True)
    if max_candidates > 0:
        return candidates[:max_candidates]
    return candidates


def _auto_max_candidates(max_sku: int, max_refs_per_sku: int, total: int) -> int:
    # Enough candidates to fill all SKU refs, but capped to keep pairwise similarity feasible.
    target = max(300, max_sku * max(1, max_refs_per_sku))
    return min(total, max(1, min(target, 2000)))


def _copy_provisional_items(
    selected: List[tuple[CropRecord, int, int]],
    provisional_dir: Path,
) -> List[ProvisionalSkuItem]:
    if provisional_dir.exists():
        shutil.rmtree(provisional_dir)
    provisional_dir.mkdir(parents=True, exist_ok=True)

    items: List[ProvisionalSkuItem] = []
    for index, (crop, width, height) in enumerate(selected, start=1):
        candidate_id = f"sku_candidate_{index:06d}"
        candidate_dir = provisional_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        dst = candidate_dir / "ref_001.jpg"
        shutil.copy2(crop.crop_path, dst)
        items.append(
            ProvisionalSkuItem(
                candidate_id=candidate_id,
                source_image=crop.image_path,
                source_object_id=crop.object_id,
                source_crop_path=crop.crop_path,
                candidate_image_path=str(dst),
                score=float(crop.score),
                width=width,
                height=height,
                source_type=crop.source_type,
                has_mask=bool(crop.has_mask),
            )
        )
    return items


def _extract_features(items: List[ProvisionalSkuItem]) -> np.ndarray:
    extractor = VisualFeatureExtractor()
    features: List[np.ndarray] = []
    valid_items: List[ProvisionalSkuItem] = []
    for item in items:
        try:
            features.append(extractor.extract_from_path(item.candidate_image_path))
            valid_items.append(item)
        except Exception:
            continue
    items[:] = valid_items
    if not features:
        return np.zeros((0, 1), dtype=np.float32)
    matrix = np.stack(features, axis=0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _centroid(feature_matrix: np.ndarray, indices: List[int]) -> np.ndarray:
    if not indices:
        return np.zeros((feature_matrix.shape[1],), dtype=np.float32)
    vector = feature_matrix[indices].mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _cluster_internal_stats(sim_matrix: np.ndarray, indices: List[int]) -> tuple[float, float]:
    if len(indices) <= 1:
        return 1.0, 1.0
    values: List[float] = []
    for pos, i in enumerate(indices):
        for j in indices[pos + 1 :]:
            values.append(float(sim_matrix[i, j]))
    if not values:
        return 1.0, 1.0
    return float(np.mean(values)), float(np.min(values))


def _merge_clusters(
    items: List[ProvisionalSkuItem],
    features: np.ndarray,
    merge_threshold: float,
    strong_merge_threshold: float,
    min_cluster_similarity: float,
    pair_report_threshold: float,
    max_refs_per_sku: int,
    out_dir: Path,
) -> tuple[List[List[int]], Path, Path]:
    n = len(items)
    pair_path = out_dir / "sku_similarity_pairs.csv"
    decisions_path = out_dir / "sku_merge_decisions.csv"
    if n == 0:
        pd.DataFrame().to_csv(pair_path, index=False)
        pd.DataFrame().to_csv(decisions_path, index=False)
        return [], pair_path, decisions_path

    sim = np.clip(features @ features.T, -1.0, 1.0)
    pair_rows: List[dict] = []
    edges: List[tuple[float, int, int]] = []
    max_pair_rows = 100_000

    for i in range(n):
        row = sim[i]
        for j in range(i + 1, n):
            similarity = float(row[j])
            if similarity >= merge_threshold:
                edges.append((similarity, i, j))
            if similarity >= pair_report_threshold and len(pair_rows) < max_pair_rows:
                pair_rows.append(
                    {
                        "candidate_a": items[i].candidate_id,
                        "candidate_b": items[j].candidate_id,
                        "similarity": similarity,
                        "similarity_percent": round(similarity * 100, 2),
                        "pair_decision": "edge" if similarity >= merge_threshold else "report_only",
                    }
                )

    pair_rows.sort(key=lambda row: row["similarity"], reverse=True)
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)

    uf = _UnionFind(n)
    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
    decisions: List[dict] = []
    max_decision_rows = 100_000

    for edge_similarity, i, j in sorted(edges, reverse=True):
        ri = uf.find(i)
        rj = uf.find(j)
        if ri == rj:
            continue

        cluster_i = clusters[ri]
        cluster_j = clusters[rj]
        combined = cluster_i + cluster_j
        if len(combined) > max_refs_per_sku:
            if len(decisions) < max_decision_rows:
                decisions.append(
                    {
                        "candidate_a": items[i].candidate_id,
                        "candidate_b": items[j].candidate_id,
                        "edge_similarity": edge_similarity,
                        "centroid_similarity": 0.0,
                        "mean_cross_similarity": 0.0,
                        "min_cross_similarity": 0.0,
                        "new_internal_mean_similarity": 0.0,
                        "new_internal_min_similarity": 0.0,
                        "decision": "reject",
                        "reason": "max_refs_per_sku_exceeded",
                    }
                )
            continue

        ci = _centroid(features, cluster_i)
        cj = _centroid(features, cluster_j)
        centroid_similarity = float(np.dot(ci, cj))
        cross_values = sim[np.ix_(cluster_i, cluster_j)].astype(np.float32)
        mean_cross = float(cross_values.mean()) if cross_values.size else 0.0
        min_cross = float(cross_values.min()) if cross_values.size else 0.0
        internal_mean, internal_min = _cluster_internal_stats(sim, combined)

        strong_rule = edge_similarity >= strong_merge_threshold and centroid_similarity >= merge_threshold - 0.02 and min_cross >= min_cluster_similarity
        normal_rule = centroid_similarity >= merge_threshold and mean_cross >= min_cluster_similarity and internal_min >= min_cluster_similarity
        accepted = bool(strong_rule or normal_rule)
        reason = "strong_rule" if strong_rule else "normal_rule" if normal_rule else "cluster_consistency_failed"

        if accepted:
            new_root = uf.union(ri, rj)
            old_root = rj if new_root == ri else ri
            clusters[new_root] = combined
            clusters.pop(old_root, None)

        if len(decisions) < max_decision_rows:
            decisions.append(
                {
                    "candidate_a": items[i].candidate_id,
                    "candidate_b": items[j].candidate_id,
                    "edge_similarity": edge_similarity,
                    "centroid_similarity": centroid_similarity,
                    "mean_cross_similarity": mean_cross,
                    "min_cross_similarity": min_cross,
                    "new_internal_mean_similarity": internal_mean,
                    "new_internal_min_similarity": internal_min,
                    "decision": "merge" if accepted else "reject",
                    "reason": reason,
                }
            )

    pd.DataFrame(decisions).to_csv(decisions_path, index=False)
    return list(clusters.values()), pair_path, decisions_path


def _copy_final_gallery(
    clusters: List[List[int]],
    items: List[ProvisionalSkuItem],
    sim_matrix: np.ndarray,
    gallery_dir: Path,
    prefix: str,
    max_sku: int,
) -> tuple[List[ClusteredDemoGalleryItem], int]:
    # Sort clusters by the best source crop score, then by cluster size.
    clusters = sorted(clusters, key=lambda idxs: (max(items[i].score for i in idxs), len(idxs)), reverse=True)
    selected_clusters = clusters[:max_sku]
    dropped_candidates = sum(len(cluster) for cluster in clusters[max_sku:])

    final_items: List[ClusteredDemoGalleryItem] = []
    for sku_number, cluster in enumerate(selected_clusters, start=1):
        sku_id = f"{prefix}{sku_number:03d}"
        sku_name = sku_id.replace("_", " ")
        sku_dir = gallery_dir / sku_id
        sku_dir.mkdir(parents=True, exist_ok=True)

        # Put the highest-score crop first.
        ordered = sorted(cluster, key=lambda index: items[index].score, reverse=True)
        mean_sim, min_sim = _cluster_internal_stats(sim_matrix, ordered)
        for ref_index, item_index in enumerate(ordered, start=1):
            item = items[item_index]
            dst = sku_dir / f"ref_{ref_index:03d}.jpg"
            shutil.copy2(item.candidate_image_path, dst)
            final_items.append(
                ClusteredDemoGalleryItem(
                    sku_id=sku_id,
                    sku_name=sku_name,
                    candidate_id=item.candidate_id,
                    source_image=item.source_image,
                    source_object_id=item.source_object_id,
                    source_crop_path=item.source_crop_path,
                    gallery_image_path=str(dst),
                    score=item.score,
                    width=item.width,
                    height=item.height,
                    source_type=item.source_type,
                    ref_index=ref_index,
                    cluster_size=len(ordered),
                    cluster_mean_similarity=mean_sim,
                    cluster_min_similarity=min_sim,
                    is_primary_ref=ref_index == 1,
                )
            )
    return final_items, dropped_candidates


def _write_cluster_outputs(
    final_items: List[ClusteredDemoGalleryItem],
    sim_matrix: np.ndarray,
    out_dir: Path,
) -> tuple[Path, Path]:
    clusters_csv = out_dir / "sku_clusters.csv"
    summary_csv = out_dir / "sku_cluster_summary.csv"
    pd.DataFrame([asdict(item) for item in final_items]).to_csv(clusters_csv, index=False)

    rows = []
    for sku_id, group in pd.DataFrame([asdict(item) for item in final_items]).groupby("sku_id") if final_items else []:
        rows.append(
            {
                "sku_id": sku_id,
                "refs_count": int(len(group)),
                "mean_internal_similarity": float(group["cluster_mean_similarity"].iloc[0]),
                "min_internal_similarity": float(group["cluster_min_similarity"].iloc[0]),
                "primary_ref": str(group.sort_values("ref_index")["gallery_image_path"].iloc[0]),
            }
        )
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    return clusters_csv, summary_csv


def _write_contact_sheets(final_items: List[ClusteredDemoGalleryItem], out_dir: Path, limit: int = 50) -> Path:
    sheets_dir = out_dir / "cluster_contact_sheets"
    if sheets_dir.exists():
        shutil.rmtree(sheets_dir)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    if not final_items:
        return sheets_dir

    df = pd.DataFrame([asdict(item) for item in final_items])
    for sku_id, group in list(df.groupby("sku_id"))[:limit]:
        images = []
        for _, row in group.sort_values("ref_index").iterrows():
            image = cv2.imread(str(row["gallery_image_path"]))
            if image is None:
                continue
            thumb = cv2.resize(image, (120, 120), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"ref_{int(row['ref_index']):03d}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            images.append(thumb)
        if not images:
            continue
        cols = min(5, len(images))
        rows = int(math.ceil(len(images) / cols))
        sheet = np.full((rows * 120, cols * 120, 3), 255, dtype=np.uint8)
        for idx, image in enumerate(images):
            y = (idx // cols) * 120
            x = (idx % cols) * 120
            sheet[y : y + 120, x : x + 120] = image
        cv2.imwrite(str(sheets_dir / f"{sku_id}.jpg"), sheet)
    return sheets_dir


def _write_report(
    out_dir: Path,
    summary: ClusteredGallerySummary,
    pair_path: Path,
    decisions_path: Path,
    clusters_csv: Path,
    cluster_summary_csv: Path,
    contact_sheets_dir: Path,
) -> Path:
    report_md = out_dir / "sku_merge_report.md"
    lines = [
        "# ShelfVision: cluster-based demo SKU gallery",
        "",
        "## Назначение",
        "",
        "В этом режиме каждый отобранный crop сначала сохраняется как отдельный provisional SKU. Затем между provisional SKU рассчитывается визуальная похожесть, после чего похожие кандидаты объединяются в финальные `sku_demo_XXX`.",
        "",
        "## Сводка",
        "",
        f"- Извлечено crop-ов: {summary.extracted_crops_count}",
        f"- Отобрано provisional SKU: {summary.provisional_sku_count}",
        f"- Итоговых SKU: {summary.created_sku_count}",
        f"- Эталонов в финальной gallery: {summary.gallery_refs_count}",
        f"- Дополнительных refs внутри SKU: {summary.duplicate_refs_count}",
        f"- Отброшено кандидатов: {summary.dropped_candidates_count}",
        f"- merge_threshold: {summary.cluster_merge_threshold}",
        f"- strong_merge_threshold: {summary.cluster_strong_merge_threshold}",
        f"- min_cluster_similarity: {summary.cluster_min_similarity}",
        "",
        "## Основные файлы",
        "",
        f"- provisional_sku_items.csv: `{out_dir / 'provisional_sku_items.csv'}`",
        f"- sku_similarity_pairs.csv: `{pair_path}`",
        f"- sku_merge_decisions.csv: `{decisions_path}`",
        f"- sku_clusters.csv: `{clusters_csv}`",
        f"- sku_cluster_summary.csv: `{cluster_summary_csv}`",
        f"- contact sheets: `{contact_sheets_dir}`",
        "",
        "## Интерпретация",
        "",
        "Новый режим уменьшает зависимость результата от порядка просмотра crop-ов. В отличие от жадной дедупликации, здесь сначала сохраняется множество кандидатов, затем анализируются пары похожих кандидатов и только после этого создаётся финальная SKU-галерея.",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return report_md


def build_clustered_demo_sku_gallery_from_predictions(
    predictions_json: str | Path,
    images_dir: str | Path | None,
    gallery_dir: str | Path,
    gallery_csv: str | Path,
    out_dir: str | Path,
    max_sku: int = 100,
    min_score: float = 0.35,
    min_width: int = 20,
    min_height: int = 20,
    use_masks: bool = True,
    padding_ratio: float = 0.05,
    prefix: str = "sku_demo_",
    clear_old_demo: bool = True,
    merge_threshold: float = 0.82,
    strong_merge_threshold: float = 0.88,
    min_cluster_similarity: float = 0.72,
    pair_report_threshold: float = 0.75,
    max_candidates: int = 0,
    max_refs_per_sku: int = 10,
) -> Dict[str, Path]:
    predictions_json = Path(predictions_json)
    gallery_dir = Path(gallery_dir)
    gallery_csv = Path(gallery_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_json.exists():
        raise FileNotFoundError(f"predictions_json не найден: {predictions_json}")

    crops_out_dir = out_dir / "demo_gallery_crops"
    crops = extract_crops_from_predictions_file(
        predictions_json=predictions_json,
        images_dir=images_dir,
        out_dir=crops_out_dir,
        use_masks=use_masks,
        padding_ratio=padding_ratio,
    )

    preliminary = _select_candidates(crops, min_score=min_score, min_width=min_width, min_height=min_height, max_candidates=0)
    effective_max_candidates = int(max_candidates) if max_candidates and max_candidates > 0 else _auto_max_candidates(max(1, max_sku), max(1, max_refs_per_sku), len(preliminary))
    selected = preliminary[:effective_max_candidates]

    if clear_old_demo:
        _clear_old_demo_skus(gallery_dir, prefix=prefix)
    else:
        gallery_dir.mkdir(parents=True, exist_ok=True)

    provisional_dir = out_dir / "provisional_skus"
    provisional_items = _copy_provisional_items(selected, provisional_dir=provisional_dir)
    features = _extract_features(provisional_items)

    clusters, pair_path, decisions_path = _merge_clusters(
        items=provisional_items,
        features=features,
        merge_threshold=merge_threshold,
        strong_merge_threshold=strong_merge_threshold,
        min_cluster_similarity=min_cluster_similarity,
        pair_report_threshold=pair_report_threshold,
        max_refs_per_sku=max(1, max_refs_per_sku),
        out_dir=out_dir,
    )

    sim_matrix = np.clip(features @ features.T, -1.0, 1.0) if len(provisional_items) else np.zeros((0, 0), dtype=np.float32)
    final_items, dropped_candidates = _copy_final_gallery(
        clusters=clusters,
        items=provisional_items,
        sim_matrix=sim_matrix,
        gallery_dir=gallery_dir,
        prefix=prefix,
        max_sku=max(1, max_sku),
    )
    clusters_csv, cluster_summary_csv = _write_cluster_outputs(final_items, sim_matrix=sim_matrix, out_dir=out_dir)
    contact_sheets_dir = _write_contact_sheets(final_items, out_dir=out_dir)

    items_json = out_dir / "demo_sku_gallery_items.json"
    items_csv = out_dir / "demo_sku_gallery_items.csv"
    provisional_csv = out_dir / "provisional_sku_items.csv"
    provisional_json = out_dir / "provisional_sku_items.json"
    pd.DataFrame([asdict(item) for item in final_items]).to_csv(items_csv, index=False)
    items_json.write_text(json.dumps([asdict(item) for item in final_items], ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(item) for item in provisional_items]).to_csv(provisional_csv, index=False)
    provisional_json.write_text(json.dumps([asdict(item) for item in provisional_items], ensure_ascii=False, indent=2), encoding="utf-8")

    created_sku_count = len({item.sku_id for item in final_items})
    duplicate_refs_count = max(0, len(final_items) - created_sku_count)
    skipped_duplicate_crops = max(0, len(selected) - len(final_items))

    status = "ok" if final_items else "error"
    warning = ""
    if not final_items:
        warning = "Не удалось сформировать cluster-based demo SKU gallery. Проверь predictions_json и фильтры crop-ов."
    elif created_sku_count < max_sku:
        status = "warning"
        warning = f"Итоговых SKU меньше max_sku: {created_sku_count} из {max_sku}."

    summary = ClusteredGallerySummary(
        predictions_json=str(predictions_json),
        images_dir=str(images_dir or ""),
        gallery_dir=str(gallery_dir),
        gallery_csv=str(gallery_csv),
        crops_dir=str(crops_out_dir / "crops"),
        provisional_dir=str(provisional_dir),
        requested_sku_count=max_sku,
        created_sku_count=created_sku_count,
        extracted_crops_count=len(crops),
        selected_crops_count=len(selected),
        provisional_sku_count=len(provisional_items),
        gallery_refs_count=len(final_items),
        duplicate_refs_count=duplicate_refs_count,
        skipped_duplicate_crops_count=skipped_duplicate_crops,
        dropped_candidates_count=dropped_candidates,
        min_score=min_score,
        min_width=min_width,
        min_height=min_height,
        use_masks=use_masks,
        deduplicate=True,
        dedup_threshold=merge_threshold,
        max_refs_per_sku=max(1, max_refs_per_sku),
        gallery_build_mode="cluster",
        cluster_merge_threshold=merge_threshold,
        cluster_strong_merge_threshold=strong_merge_threshold,
        cluster_min_similarity=min_cluster_similarity,
        cluster_pair_report_threshold=pair_report_threshold,
        cluster_max_candidates=effective_max_candidates,
        status=status,
        warning=warning,
    )
    summary_json = out_dir / "demo_sku_gallery_summary.json"
    summary_json.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    gallery_outputs: Dict[str, Path] = {}
    if final_items:
        gallery_outputs = build_sku_gallery(
            gallery_dir=gallery_dir,
            output_csv=gallery_csv,
            out_dir=out_dir / "gallery_check",
            min_images_per_sku=1,
        )

    report_md = _write_report(
        out_dir=out_dir,
        summary=summary,
        pair_path=pair_path,
        decisions_path=decisions_path,
        clusters_csv=clusters_csv,
        cluster_summary_csv=cluster_summary_csv,
        contact_sheets_dir=contact_sheets_dir,
    )
    # Keep the old filename so the existing Control Panel tab can render the report.
    demo_report_md = out_dir / "demo_sku_gallery_report.md"
    shutil.copyfile(report_md, demo_report_md)

    outputs = {
        "summary_json": summary_json,
        "items_json": items_json,
        "items_csv": items_csv,
        "provisional_csv": provisional_csv,
        "provisional_json": provisional_json,
        "similarity_pairs_csv": pair_path,
        "merge_decisions_csv": decisions_path,
        "clusters_csv": clusters_csv,
        "cluster_summary_csv": cluster_summary_csv,
        "contact_sheets_dir": contact_sheets_dir,
        "report_md": report_md,
        "demo_report_md": demo_report_md,
    }
    outputs.update(gallery_outputs)
    return outputs
