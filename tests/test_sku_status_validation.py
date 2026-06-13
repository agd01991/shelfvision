from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.validate_run_outputs import validate_run


def test_validator_counts_real_sku_status_column(tmp_path: Path) -> None:
    inference_dir = tmp_path / "01_inference"
    gallery_dir = tmp_path / "02_demo_gallery" / "sku_gallery_final"
    identification_dir = tmp_path / "04_identification"
    visualized_dir = identification_dir / "visualized"
    crops_dir = tmp_path / "02_demo_gallery"

    inference_dir.mkdir(parents=True)
    gallery_dir.mkdir(parents=True)
    identification_dir.mkdir(parents=True)
    visualized_dir.mkdir(parents=True)

    crop_path = crops_dir / "crop_001.jpg"
    crop_path.write_text("demo", encoding="utf-8")
    (visualized_dir / "sample.jpg").write_text("demo", encoding="utf-8")
    (inference_dir / "predictions.json").write_text(json.dumps([{"image_path": "sample.jpg"}]), encoding="utf-8")
    pd.DataFrame([{"image_path": "sample.jpg", "objects_count": 3}]).to_csv(inference_dir / "summary.csv", index=False)
    pd.DataFrame([{"crop_path": str(crop_path), "object_id": 1}]).to_csv(crops_dir / "crops_manifest.csv", index=False)
    pd.DataFrame([{"sku_id": "sku_001", "image_path": str(crop_path)}]).to_csv(gallery_dir / "gallery.csv", index=False)
    pd.DataFrame(
        [
            {"object_id": 1, "sku_status": "matched", "sku_confidence": 0.9},
            {"object_id": 2, "sku_status": "matched_uncertain", "sku_confidence": 0.8},
            {"object_id": 3, "sku_status": "unknown", "sku_confidence": 0.2},
        ]
    ).to_csv(identification_dir / "identification_results.csv", index=False)

    summary, checks, artifacts = validate_run(tmp_path)

    assert summary.identification_rows == 3
    assert summary.matched == 1
    assert summary.matched_uncertain == 1
    assert summary.unknown == 1
    assert artifacts["identification_results_csv"].endswith("identification_results.csv")
