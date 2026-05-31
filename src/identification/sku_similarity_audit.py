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


@dataclass
class SkuAuditSummary:
    gallery_dir: str
    out_dir: str
    sku_count: int
    refs_count: int
    usable_sku_count: int
    usable_refs_count: int
    pairs_reported_count: int
    merge_candidates_count: int
    pair_report_threshold: float
    candidate_threshold: float
    top_n: int
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
    return sorted(path for path in sku_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _safe_pair_name(sku_a: str, sku_b: str) -> str:
    safe_a = re.sub(r"[^A-Za-z0-9_.-]+", "_", sku_a)
    safe_b = re.sub(r"[^A-Za-z0-9_.-]+", "_", sku_b)
    return f"{safe_a}__{safe_b}.jpg"


def _extract_gallery_features(
    gallery_dir: Path,
    max_refs_per_sku: int = 10,
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

        features: List[np.ndarray] = []
        usable_refs: List[Path] = []
        for ref in refs:
            try:
                feature = extractor.extract_from_path(ref)
            except Exception:
                continue
            features.append(feature.astype(np.float32))
            usable_refs.append(ref)

        if not features:
            continue
        matrix = np.stack(features, axis=0).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        refs_by_sku[sku_id] = usable_refs
        features_by_sku[sku_id] = matrix
        usable_refs_count += len(usable_refs)

    return refs_by_sku, features_by_sku, refs_count, usable_refs_count


def _centroid(matrix: np.ndarray) -> np.ndarray:
    vector = matrix.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _pair_scores(features_a: np.ndarray, features_b: np.ndarray) -> dict:
    centroid_a = _centroid(features_a)
    centroid_b = _centroid(features_b)
    pair_matrix = np.clip(features_a @ features_b.T, -1.0, 1.0)
    return {
        "centroid_similarity": float(np.dot(centroid_a, centroid_b)),
        "best_pair_similarity": float(pair_matrix.max()),
        "mean_pair_similarity": float(pair_matrix.mean()),
        "min_pair_similarity": float(pair_matrix.min()),
    }


def _recommendation(scores: dict, candidate_threshold: float) -> str:
    centroid = float(scores["centroid_similarity"])
    best = float(scores["best_pair_similarity"])
    mean = float(scores["mean_pair_similarity"])
    if centroid >= candidate_threshold and mean >= candidate_threshold - 0.04:
        return "merge_candidate"
    if best >= candidate_threshold + 0.05 and centroid >= candidate_threshold - 0.05:
        return "review_candidate"
    return "similar_pair"


def run_sku_similarity_audit(
    gallery_dir: str | Path,
    out_dir: str | Path,
    pair_report_threshold: float = 0.75,
    candidate_threshold: float = 0.82,
    top_n: int = 100,
    contact_sheet_limit: int = 50,
    max_refs_per_sku: int = 10,
) -> Dict[str, Path]:
    gallery_dir = _current_os_path(gallery_dir)
    out_dir = _current_os_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets_dir = out_dir / "sku_pair_contact_sheets"
    contact_sheets_dir.mkdir(parents=True, exist_ok=True)

    refs_by_sku, features_by_sku, refs_count, usable_refs_count = _extract_gallery_features(
        gallery_dir=gallery_dir,
        max_refs_per_sku=max_refs_per_sku,
    )

    sku_ids = sorted(features_by_sku.keys())
    rows: List[dict] = []
    for i, sku_a in enumerate(sku_ids):
        for sku_b in sku_ids[i + 1:]:
            scores = _pair_scores(features_by_sku[sku_a], features_by_sku[sku_b])
            report_score = max(
                scores["centroid_similarity"],
                scores["best_pair_similarity"],
                scores["mean_pair_similarity"],
            )
            if report_score < pair_report_threshold:
                continue
            recommendation = _recommendation(scores, candidate_threshold=candidate_threshold)
            rows.append({
                "sku_a": sku_a,
                "sku_b": sku_b,
                **scores,
                "recommendation": recommendation,
                "sku_a_refs": len(refs_by_sku.get(sku_a, [])),
                "sku_b_refs": len(refs_by_sku.get(sku_b, [])),
                "pair_contact_sheet": str(contact_sheets_dir / _safe_pair_name(sku_a, sku_b)),
            })

    rows.sort(
        key=lambda row: (
            row["recommendation"] == "merge_candidate",
            row["recommendation"] == "review_candidate",
            row["centroid_similarity"],
            row["best_pair_similarity"],
        ),
        reverse=True,
    )
    rows_to_write = rows[:top_n] if top_n > 0 else rows
    candidate_rows = [
        row for row in rows_to_write
        if row["recommendation"] in {"merge_candidate", "review_candidate"}
    ]

    pairs_csv = out_dir / "sku_to_sku_similarity.csv"
    candidates_csv = out_dir / "merge_candidates.csv"
    pd.DataFrame(rows_to_write).to_csv(pairs_csv, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)

    summary = SkuAuditSummary(
        gallery_dir=str(gallery_dir),
        out_dir=str(out_dir),
        sku_count=len(list(_iter_sku_dirs(gallery_dir))),
        refs_count=refs_count,
        usable_sku_count=len(sku_ids),
        usable_refs_count=usable_refs_count,
        pairs_reported_count=len(rows_to_write),
        merge_candidates_count=len(candidate_rows),
        pair_report_threshold=pair_report_threshold,
        candidate_threshold=candidate_threshold,
        top_n=top_n,
        status="ok" if sku_ids else "error",
    )
    summary_json = out_dir / "sku_similarity_audit_summary.json"
    summary_json.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = out_dir / "sku_similarity_audit_report.md"
    lines = [
        "# ShelfVision: SKU-to-SKU similarity audit",
        "",
        "## Назначение",
        "",
        "Модуль анализирует уже сформированную SKU-галерею и ищет пары SKU-папок, визуально похожие друг на друга. Такие пары можно использовать как кандидаты для ручного merge в Control Panel.",
        "",
        "## Сводка",
        "",
        f"- Gallery dir: `{summary.gallery_dir}`",
        f"- SKU всего: {summary.sku_count}",
        f"- SKU с доступными признаками: {summary.usable_sku_count}",
        f"- Refs всего: {summary.refs_count}",
        f"- Refs с доступными признаками: {summary.usable_refs_count}",
        f"- Пары в отчёте: {summary.pairs_reported_count}",
        f"- Merge/review candidates: {summary.merge_candidates_count}",
        f"- Pair report threshold: {summary.pair_report_threshold}",
        f"- Candidate threshold: {summary.candidate_threshold}",
        "",
        "## Основные файлы",
        "",
        f"- `sku_to_sku_similarity.csv`: `{pairs_csv}`",
        f"- `merge_candidates.csv`: `{candidates_csv}`",
        f"- `sku_pair_contact_sheets/`: `{contact_sheets_dir}`",
        "",
        "## Формулировка для ВКР",
        "",
        "После автоматического формирования SKU-галереи выполняется дополнительный аудит похожести между SKU. Для каждой пары SKU рассчитываются centroid similarity, best pair similarity и mean pair similarity. Пары с высокой похожестью предлагаются пользователю как кандидаты для ручного объединения.",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "summary_json": summary_json,
        "pairs_csv": pairs_csv,
        "candidates_csv": candidates_csv,
        "report_md": report_md,
        "contact_sheets_dir": contact_sheets_dir,
    }
