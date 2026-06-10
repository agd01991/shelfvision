from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.identification.feature_extractor import VisualFeatureExtractor
from src.identification.sku_gallery import IMAGE_EXTS


WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
WSL_MOUNT_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")

DECISION_LABELS_RU = {
    "ok": "эталон согласуется со своим SKU",
    "possible_outlier": "возможный выброс внутри SKU",
    "likely_wrong_sku": "вероятно относится к другому SKU",
}


@dataclass
class SkuPurityAuditSummary:
    gallery_dir: str
    out_dir: str
    sku_count: int
    refs_count: int
    usable_sku_count: int
    usable_refs_count: int
    checked_refs_count: int
    ok_refs_count: int
    possible_outliers_count: int
    likely_wrong_sku_count: int
    mixed_sku_count: int
    own_centroid_threshold: float
    own_mean_threshold: float
    other_margin: float
    min_other_similarity: float
    max_refs_per_sku: int
    status: str


def _current_os_path(value: str | Path | None) -> Path:
    raw = str(value or "").strip().strip('"').strip("'").replace("\\", "/")
    if os.name == "nt":
        match = WSL_MOUNT_RE.match(raw)
        if match:
            return Path(f"{match.group(1).upper()}:/{match.group(2)}")
        return Path(raw)

    match = WINDOWS_DRIVE_RE.match(raw)
    if match:
        return Path(f"/mnt/{match.group(1).lower()}/{match.group(2)}")
    return Path(raw)


def _iter_sku_dirs(gallery_dir: Path) -> Iterable[Path]:
    if not gallery_dir.exists():
        return []
    return sorted(path for path in gallery_dir.iterdir() if path.is_dir())


def _iter_refs(sku_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in sku_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _centroid(matrix: np.ndarray) -> np.ndarray:
    vector = matrix.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _extract_features(
    gallery_dir: Path,
    max_refs_per_sku: int,
) -> tuple[Dict[str, List[Path]], Dict[str, np.ndarray], int, int]:
    extractor = VisualFeatureExtractor()

    refs_by_sku: Dict[str, List[Path]] = {}
    features_by_sku: Dict[str, np.ndarray] = {}

    refs_count = 0
    usable_refs_count = 0

    for sku_dir in _iter_sku_dirs(gallery_dir):
        sku_id = sku_dir.name
        refs = _iter_refs(sku_dir)
        refs_count += len(refs)

        if max_refs_per_sku > 0:
            refs = refs[:max_refs_per_sku]

        usable_refs: List[Path] = []
        features: List[np.ndarray] = []

        for ref in refs:
            try:
                feature = extractor.extract_from_path(ref)
            except Exception:
                continue

            usable_refs.append(ref)
            features.append(feature.astype(np.float32))

        if not features:
            continue

        matrix = np.stack(features, axis=0).astype(np.float32)
        matrix = _normalize_matrix(matrix)

        refs_by_sku[sku_id] = usable_refs
        features_by_sku[sku_id] = matrix
        usable_refs_count += len(usable_refs)

    return refs_by_sku, features_by_sku, refs_count, usable_refs_count


def _own_similarity_stats(matrix: np.ndarray, index: int) -> dict:
    feature = matrix[index]

    if len(matrix) == 1:
        return {
            "own_centroid_similarity": 1.0,
            "own_mean_similarity": 1.0,
            "own_min_similarity": 1.0,
        }

    other_indices = [i for i in range(len(matrix)) if i != index]
    other_matrix = matrix[other_indices]

    own_centroid = _centroid(other_matrix)
    pair_scores = np.clip(other_matrix @ feature, -1.0, 1.0)

    return {
        "own_centroid_similarity": float(np.dot(feature, own_centroid)),
        "own_mean_similarity": float(pair_scores.mean()),
        "own_min_similarity": float(pair_scores.min()),
    }


def _build_sku_centroids(features_by_sku: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {sku_id: _centroid(matrix) for sku_id, matrix in features_by_sku.items()}


def _nearest_other_sku(
    feature: np.ndarray,
    source_sku_id: str,
    centroids_by_sku: Dict[str, np.ndarray],
) -> tuple[str, float]:
    best_sku = ""
    best_score = -1.0

    for sku_id, centroid in centroids_by_sku.items():
        if sku_id == source_sku_id:
            continue

        score = float(np.dot(feature, centroid))
        if score > best_score:
            best_score = score
            best_sku = sku_id

    return best_sku, best_score


def _decision(
    own_centroid_similarity: float,
    own_mean_similarity: float,
    nearest_other_similarity: float,
    own_centroid_threshold: float,
    own_mean_threshold: float,
    other_margin: float,
    min_other_similarity: float,
) -> str:
    if (
        nearest_other_similarity >= min_other_similarity
        and nearest_other_similarity - own_centroid_similarity >= other_margin
    ):
        return "likely_wrong_sku"

    if (
        own_centroid_similarity < own_centroid_threshold
        or own_mean_similarity < own_mean_threshold
    ):
        return "possible_outlier"

    return "ok"


def _decision_label(value: str) -> str:
    return DECISION_LABELS_RU.get(str(value), str(value))


def run_sku_purity_audit(
    gallery_dir: str | Path,
    out_dir: str | Path,
    own_centroid_threshold: float = 0.65,
    own_mean_threshold: float = 0.60,
    other_margin: float = 0.08,
    min_other_similarity: float = 0.68,
    max_refs_per_sku: int = 50,
) -> Dict[str, Path]:
    gallery_dir = _current_os_path(gallery_dir)
    out_dir = _current_os_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs_by_sku, features_by_sku, refs_count, usable_refs_count = _extract_features(
        gallery_dir=gallery_dir,
        max_refs_per_sku=max_refs_per_sku,
    )

    centroids_by_sku = _build_sku_centroids(features_by_sku)

    rows: List[dict] = []

    for sku_id, matrix in features_by_sku.items():
        refs = refs_by_sku.get(sku_id, [])

        for index, ref in enumerate(refs):
            feature = matrix[index]

            own_stats = _own_similarity_stats(matrix, index)
            nearest_sku, nearest_score = _nearest_other_sku(
                feature=feature,
                source_sku_id=sku_id,
                centroids_by_sku=centroids_by_sku,
            )

            own_centroid_similarity = float(own_stats["own_centroid_similarity"])
            own_mean_similarity = float(own_stats["own_mean_similarity"])

            decision = _decision(
                own_centroid_similarity=own_centroid_similarity,
                own_mean_similarity=own_mean_similarity,
                nearest_other_similarity=nearest_score,
                own_centroid_threshold=own_centroid_threshold,
                own_mean_threshold=own_mean_threshold,
                other_margin=other_margin,
                min_other_similarity=min_other_similarity,
            )

            rows.append(
                {
                    "sku_id": sku_id,
                    "ref_file": ref.name,
                    "ref_path": str(ref),
                    "own_centroid_similarity": own_centroid_similarity,
                    "own_mean_similarity": own_mean_similarity,
                    "own_min_similarity": float(own_stats["own_min_similarity"]),
                    "nearest_other_sku": nearest_sku,
                    "nearest_other_similarity": nearest_score,
                    "other_minus_own": nearest_score - own_centroid_similarity,
                    "decision": decision,
                    "suggested_new_sku_id": f"{sku_id}_split_{index + 1:03d}",
                }
            )

    ref_df = pd.DataFrame(rows)
    ref_purity_csv = out_dir / "sku_ref_purity.csv"
    ref_df.to_csv(ref_purity_csv, index=False)

    if ref_df.empty:
        outliers_df = pd.DataFrame()
        mixed_df = pd.DataFrame()
    else:
        outliers_df = ref_df[ref_df["decision"].isin(["possible_outlier", "likely_wrong_sku"])].copy()

        mixed_rows = []
        for sku_id, group in ref_df.groupby("sku_id"):
            refs_total = len(group)
            outliers_count = int(group["decision"].isin(["possible_outlier", "likely_wrong_sku"]).sum())
            likely_wrong_count = int(group["decision"].eq("likely_wrong_sku").sum())
            possible_count = int(group["decision"].eq("possible_outlier").sum())

            if outliers_count == 0:
                continue

            mixed_rows.append(
                {
                    "sku_id": sku_id,
                    "refs_total": refs_total,
                    "outliers_count": outliers_count,
                    "likely_wrong_sku_count": likely_wrong_count,
                    "possible_outliers_count": possible_count,
                    "outlier_rate": outliers_count / max(1, refs_total),
                    "mean_own_centroid_similarity": float(group["own_centroid_similarity"].mean()),
                    "min_own_centroid_similarity": float(group["own_centroid_similarity"].min()),
                    "max_nearest_other_similarity": float(group["nearest_other_similarity"].max()),
                }
            )

        mixed_df = pd.DataFrame(mixed_rows)

    outliers_csv = out_dir / "ref_outlier_candidates.csv"
    mixed_csv = out_dir / "mixed_sku_candidates.csv"

    outliers_df.to_csv(outliers_csv, index=False)
    mixed_df.to_csv(mixed_csv, index=False)

    checked_refs_count = len(ref_df)
    ok_refs_count = int(ref_df["decision"].eq("ok").sum()) if not ref_df.empty else 0
    possible_outliers_count = int(ref_df["decision"].eq("possible_outlier").sum()) if not ref_df.empty else 0
    likely_wrong_sku_count = int(ref_df["decision"].eq("likely_wrong_sku").sum()) if not ref_df.empty else 0
    mixed_sku_count = int(len(mixed_df)) if not mixed_df.empty else 0

    summary = SkuPurityAuditSummary(
        gallery_dir=str(gallery_dir),
        out_dir=str(out_dir),
        sku_count=len(list(_iter_sku_dirs(gallery_dir))),
        refs_count=refs_count,
        usable_sku_count=len(features_by_sku),
        usable_refs_count=usable_refs_count,
        checked_refs_count=checked_refs_count,
        ok_refs_count=ok_refs_count,
        possible_outliers_count=possible_outliers_count,
        likely_wrong_sku_count=likely_wrong_sku_count,
        mixed_sku_count=mixed_sku_count,
        own_centroid_threshold=own_centroid_threshold,
        own_mean_threshold=own_mean_threshold,
        other_margin=other_margin,
        min_other_similarity=min_other_similarity,
        max_refs_per_sku=max_refs_per_sku,
        status="ok" if checked_refs_count else "error",
    )

    summary_json = out_dir / "sku_purity_audit_summary.json"
    summary_json.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = out_dir / "sku_purity_audit_report.md"
    lines = [
        "# ShelfVision: аудит чистоты SKU-галереи",
        "",
        "## Назначение",
        "",
        "Модуль проверяет чистоту уже сформированной SKU-галереи и ищет случаи, когда разные товары могли попасть в один `sku_id`.",
        "",
        "## Сводка",
        "",
        f"- Папка галереи: `{summary.gallery_dir}`",
        f"- SKU всего: {summary.sku_count}",
        f"- SKU с доступными признаками: {summary.usable_sku_count}",
        f"- Эталонов всего: {summary.refs_count}",
        f"- Эталонов с доступными признаками: {summary.usable_refs_count}",
        f"- Эталонов проверено: {summary.checked_refs_count}",
        f"- Эталонов без признаков ошибки: {summary.ok_refs_count}",
        f"- Возможных выбросов внутри SKU: {summary.possible_outliers_count}",
        f"- Эталонов, вероятно относящихся к другому SKU: {summary.likely_wrong_sku_count}",
        f"- Кандидатов на смешанные SKU: {summary.mixed_sku_count}",
        "",
        "## Пороговые параметры",
        "",
        f"- Порог сходства с центром своего SKU: {summary.own_centroid_threshold}",
        f"- Порог среднего сходства внутри своего SKU: {summary.own_mean_threshold}",
        f"- Минимальный отрыв от ближайшего другого SKU: {summary.other_margin}",
        f"- Минимальное сходство с другим SKU: {summary.min_other_similarity}",
        f"- Максимум эталонов на один SKU: {summary.max_refs_per_sku}",
        "",
        "## Типы решений",
        "",
        "| Машинное значение | Отображение |",
        "|---|---|",
        *[f"| `{key}` | {_decision_label(key)} |" for key in ["ok", "possible_outlier", "likely_wrong_sku"]],
        "",
        "## Основные файлы",
        "",
        f"- Полная таблица проверки эталонов: `{ref_purity_csv}`",
        f"- Эталоны-кандидаты на вынос в другой SKU: `{outliers_csv}`",
        f"- Кандидаты на смешанные SKU: `{mixed_csv}`",
        f"- JSON-сводка аудита: `{summary_json}`",
        "",
        "## Интерпретация",
        "",
        "`possible_outlier` означает, что эталон слабо похож на остальные эталоны своего SKU. "
        "`likely_wrong_sku` означает, что эталон больше похож на другой SKU, чем на собственный.",
        "",
        "## Формулировка для ВКР",
        "",
        "Для выявления ошибок, при которых разные товары попадают в один SKU-кластер, был реализован аудит чистоты SKU. "
        "Для каждого эталонного изображения рассчитывается сходство с центроидом собственного SKU и ближайшим другим SKU. "
        "Изображения с низкой внутренней похожестью или более высокой похожестью к другому SKU предлагаются пользователю как кандидаты на разделение кластера.",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "ref_purity_csv": ref_purity_csv,
        "outliers_csv": outliers_csv,
        "mixed_csv": mixed_csv,
        "summary_json": summary_json,
        "report_md": report_md,
    }
