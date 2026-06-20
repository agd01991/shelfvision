from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class DefenseExportSummary:
    experiment_dir: str
    output_zip: str
    files_added: int
    skipped_missing: int
    status: str
    note: str = ""


DEFAULT_FILES = [
    "00_manifest/all_images.csv",
    "00_manifest/gallery_images.csv",
    "00_manifest/query_images.csv",
    "01_gallery_inference/predictions.json",
    "01_gallery_inference/summary.csv",
    "02_demo_gallery/demo_sku_gallery_summary.json",
    "02_demo_gallery/demo_sku_gallery_items.csv",
    "02_demo_gallery/demo_sku_gallery_report.md",
    "03_query_inference/predictions.json",
    "03_query_inference/summary.csv",
    "04_identification/crops_manifest.csv",
    "04_identification/crops_manifest.json",
    "04_identification/identification_results.csv",
    "04_identification/identification_report.md",
    "04_identification/identification_metrics.csv",
    "04_identification/assignment_uncertainty_report.md",
    "04_identification/assignment_uncertainty_summary.json",
    "04_identification/matched_uncertain_candidates.csv",
    "05_reports/full_experiment_summary.json",
    "05_reports/full_experiment_summary.csv",
    "05_reports/full_experiment_summary.md",
    "05_reports/threshold_analysis.csv",
    "05_reports/segmentation_identification_report.md",
    "06_manual_identification/manual_identification_edits.csv",
    "06_manual_identification/manual_reference_suggestions.csv",
    "06_manual_identification/identification_results_corrected.csv",
    "06_manual_identification/manual_identification_summary.json",
    "06_manual_identification/manual_identification_report.md",
    "06_manual_gallery/manual_cluster_edits.csv",
    "06_manual_gallery/manual_gallery_summary.json",
    "06_manual_gallery/manual_gallery_report.md",
    "07_sku_audit/sku_similarity_audit_report.md",
    "07_sku_audit/merge_candidates.csv",
    "07_sku_purity_audit/sku_purity_audit_report.md",
    "07_sku_purity_audit/mixed_sku_candidates.csv",
    "07_sku_purity_audit/ref_outlier_candidates.csv",
    "selected_sku_demo/selected_skus.csv",
    "selected_sku_demo/selected_identification_results.csv",
    "selected_sku_demo/selected_sku_report.md",
]

DEFAULT_DIRS = [
    "selected_sku_demo",
    "06_manual_identification/proposed_refs",
]

VISUAL_DIRS = [
    "01_gallery_inference/visualized",
    "03_query_inference/visualized",
    "04_identification/visualized",
]


def _iter_dir_files(root: Path, limit: int = 0) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if limit and limit > 0:
        files = files[:limit]
    return files


def build_defense_export_zip(
    experiment_dir: str | Path,
    output_zip: str | Path | None = None,
    include_visualizations: bool = True,
    visualized_limit_per_dir: int = 30,
) -> Dict[str, Path]:
    """Create a compact ZIP archive with key VKR defense artifacts.

    The archive intentionally contains reports, CSV/JSON summaries and a limited
    number of visualization images. Large raw datasets and model weights are not
    included.
    """

    exp = Path(experiment_dir)
    if output_zip is None:
        output_zip = exp / "defense_export" / "vkr_defense_artifacts.zip"
    output = Path(output_zip)
    output.parent.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped = 0
    added_rel: set[str] = set()

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in DEFAULT_FILES:
            path = exp / rel
            if path.exists() and path.is_file():
                zf.write(path, arcname=rel)
                added += 1
                added_rel.add(rel)
            else:
                skipped += 1

        for rel_dir in DEFAULT_DIRS:
            for path in _iter_dir_files(exp / rel_dir):
                rel = str(path.relative_to(exp)).replace("\\", "/")
                if rel not in added_rel:
                    zf.write(path, arcname=rel)
                    added += 1
                    added_rel.add(rel)

        if include_visualizations:
            for rel_dir in VISUAL_DIRS:
                for path in _iter_dir_files(exp / rel_dir, limit=max(0, int(visualized_limit_per_dir))):
                    rel = str(path.relative_to(exp)).replace("\\", "/")
                    if rel not in added_rel:
                        zf.write(path, arcname=rel)
                        added += 1
                        added_rel.add(rel)

        manifest = {
            "experiment_dir": str(exp),
            "files_added": added,
            "skipped_missing": skipped,
            "include_visualizations": include_visualizations,
            "visualized_limit_per_dir": visualized_limit_per_dir,
            "note": "Raw datasets and model weights are intentionally not included.",
        }
        zf.writestr("EXPORT_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        added += 1

    status = "ok" if added > 1 else "warning"
    summary = DefenseExportSummary(
        experiment_dir=str(exp),
        output_zip=str(output),
        files_added=added,
        skipped_missing=skipped,
        status=status,
        note="Raw datasets and model weights are intentionally not included.",
    )
    summary_json = output.parent / "defense_export_summary.json"
    summary_json.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = output.parent / "defense_export_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# Экспорт материалов для защиты",
                "",
                f"- Папка эксперимента: `{summary.experiment_dir}`",
                f"- ZIP-архив: `{summary.output_zip}`",
                f"- Добавлено файлов: {summary.files_added}",
                f"- Не найдено ожидаемых файлов: {summary.skipped_missing}",
                f"- Статус: **{summary.status}**",
                "",
                "Сырые датасеты и веса моделей в архив не включаются из-за размера и лицензионных ограничений.",
            ]
        ),
        encoding="utf-8",
    )

    return {"zip": output, "summary_json": summary_json, "report_md": report_md}
