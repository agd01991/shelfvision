from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.export_presentation_assets import export_assets
from scripts.run_metadata import collect_run_counts


def _make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    inference_dir = run_dir / "03_query_inference"
    gallery_dir = run_dir / "02_demo_gallery" / "sku_gallery_final"
    identification_dir = run_dir / "04_identification"
    visualized_dir = identification_dir / "visualized"

    inference_dir.mkdir(parents=True)
    gallery_dir.mkdir(parents=True)
    identification_dir.mkdir(parents=True)
    visualized_dir.mkdir(parents=True)

    pd.DataFrame([{"image_path": "shelf.jpg", "objects_count": 3}]).to_csv(inference_dir / "summary.csv", index=False)
    pd.DataFrame([{"sku_id": "sku_1", "image_path": "ref.jpg"}]).to_csv(gallery_dir / "gallery.csv", index=False)
    pd.DataFrame([{"crop_path": "crop.jpg"}]).to_csv(identification_dir / "crops_manifest.csv", index=False)
    pd.DataFrame(
        [
            {"object_id": 1, "sku_status": "matched", "sku_confidence": 0.90, "distinct_margin": 0.10},
            {"object_id": 2, "sku_status": "matched_uncertain", "sku_confidence": 0.80, "distinct_margin": 0.02},
            {"object_id": 3, "sku_status": "unknown", "sku_confidence": 0.10, "distinct_margin": 0.00},
        ]
    ).to_csv(identification_dir / "identification_results.csv", index=False)
    pd.DataFrame(
        [
            {"object_id": 1, "sku_status": "matched", "sku_confidence": 0.90, "distinct_margin": 0.10},
            {"object_id": 2, "sku_status": "matched", "sku_confidence": 0.80, "distinct_margin": 0.02},
            {"object_id": 3, "sku_status": "unknown", "sku_confidence": 0.10, "distinct_margin": 0.00},
        ]
    ).to_csv(identification_dir / "identification_results_corrected.csv", index=False)
    (visualized_dir / "sample.jpg").write_text("demo", encoding="utf-8")
    return run_dir


def test_run_metadata_counts_sku_status(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)

    counts = collect_run_counts(run_dir)

    assert counts["status_column"] == "sku_status"
    assert counts["matched"] == 1
    assert counts["matched_uncertain"] == 1
    assert counts["unknown"] == 1
    assert counts["avg_similarity"] > 0
    assert counts["avg_margin"] > 0


def test_export_assets_counts_sku_status_and_corrected_results(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "exported"

    outputs = export_assets(run_dir, out_dir, limit=1)
    metrics = pd.read_csv(outputs["key_metrics_csv"])
    distribution = pd.read_csv(outputs["status_distribution_csv"])

    assert int(metrics.loc[0, "matched"]) == 1
    assert int(metrics.loc[0, "matched_uncertain"]) == 1
    assert int(metrics.loc[0, "unknown"]) == 1
    assert int(metrics.loc[0, "corrected_matched"]) == 2
    assert int(metrics.loc[0, "corrected_matched_uncertain"]) == 0
    assert set(distribution["status"]) == {"matched", "matched_uncertain", "unknown"}
    assert (out_dir / "identification_results_corrected.csv").exists()
