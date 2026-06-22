from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import action_history
from src.identification.feature_extractor import VisualFeatureExtractor
from src.identification.manual_identification_editor import (
    ManualIdentificationEdit,
    append_manual_identification_edit,
    apply_manual_identification_edits,
)
from src.identification.matcher import (
    SkuCandidate,
    _distinct_top2,
    _resolve_assignment_status,
)
from src.reporting.defense_export import build_defense_export_zip
from verify_experiment_source import build_report as build_source_report


class MatchingLogicTests(unittest.TestCase):
    def test_status_resolution(self) -> None:
        best = SkuCandidate("sku_a", "A", "", 0.80, "a.jpg")
        self.assertEqual(
            _resolve_assignment_status(best, 0.65, True, 0.03, 0.10),
            "matched",
        )
        self.assertEqual(
            _resolve_assignment_status(best, 0.65, True, 0.03, 0.01),
            "matched_uncertain",
        )
        weak = SkuCandidate("sku_a", "A", "", 0.60, "a.jpg")
        self.assertEqual(
            _resolve_assignment_status(weak, 0.65, True, 0.03, 0.10),
            "unknown",
        )

    def test_margin_uses_distinct_sku(self) -> None:
        candidates = [
            SkuCandidate("sku_a", "A", "", 0.90, "a1.jpg"),
            SkuCandidate("sku_a", "A", "", 0.88, "a2.jpg"),
            SkuCandidate("sku_b", "B", "", 0.81, "b1.jpg"),
        ]
        first, first_score, second, second_score, margin = _distinct_top2(candidates)
        self.assertEqual(first, "sku_a")
        self.assertEqual(second, "sku_b")
        self.assertAlmostEqual(first_score, 0.90)
        self.assertAlmostEqual(second_score, 0.81)
        self.assertAlmostEqual(float(margin), 0.09)

    def test_feature_vector_size_and_norm(self) -> None:
        extractor = VisualFeatureExtractor()
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        vector = extractor.extract_from_image(image)
        self.assertEqual(vector.shape, (2112,))
        self.assertEqual(extractor.vector_size, 2112)
        self.assertLessEqual(float(np.linalg.norm(vector)), 1.000001)


class ManualEditTests(unittest.TestCase):
    def test_latest_edit_is_applied_without_overwriting_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "identification_results.csv"
            edits = root / "manual_edits.csv"
            corrected = root / "corrected.csv"

            pd.DataFrame(
                [
                    {
                        "image_name": "image.jpg",
                        "object_id": 1,
                        "sku_id": "sku_old",
                        "sku_name": "Old",
                        "sku_status": "matched_uncertain",
                    }
                ]
            ).to_csv(source, index=False)

            append_manual_identification_edit(
                edits,
                ManualIdentificationEdit(
                    edit_id="first",
                    image_name="image.jpg",
                    image_path="image.jpg",
                    object_id=1,
                    crop_path="crop.jpg",
                    old_sku_id="sku_old",
                    old_sku_name="Old",
                    old_status="matched_uncertain",
                    old_score=0.70,
                    old_margin=0.01,
                    new_sku_id="sku_a",
                    new_sku_name="A",
                    new_status="matched",
                    edit_type="change_sku",
                ),
            )
            append_manual_identification_edit(
                edits,
                ManualIdentificationEdit(
                    edit_id="second",
                    image_name="image.jpg",
                    image_path="image.jpg",
                    object_id=1,
                    crop_path="crop.jpg",
                    old_sku_id="sku_old",
                    old_sku_name="Old",
                    old_status="matched_uncertain",
                    old_score=0.70,
                    old_margin=0.01,
                    new_sku_id="sku_b",
                    new_sku_name="B",
                    new_status="matched",
                    edit_type="change_sku",
                ),
            )

            apply_manual_identification_edits(source, edits, corrected)
            source_df = pd.read_csv(source).fillna("")
            corrected_df = pd.read_csv(corrected).fillna("")
            self.assertEqual(source_df.loc[0, "sku_id"], "sku_old")
            self.assertEqual(corrected_df.loc[0, "sku_id"], "sku_b")
            self.assertTrue(bool(corrected_df.loc[0, "manual_edit_applied"]))


class HistoryTests(unittest.TestCase):
    def test_event_and_checkpoint_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            action_history.append_event(exp, "test", "Тестовое событие", "details")
            checkpoint = action_history.create_checkpoint(
                exp,
                "Этап 1",
                config={"threshold": 0.65},
                note="ok",
            )
            events = action_history.read_events(exp)
            data = action_history.read_checkpoint(checkpoint)
            self.assertGreaterEqual(len(events), 2)
            self.assertEqual(data["title"], "Этап 1")
            self.assertEqual(data["config"]["threshold"], 0.65)


class DataSourceTests(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path, config_images_dir: str, run_images_dir: str) -> tuple[Path, Path]:
        exp = root / "experiment"
        manifest_dir = exp / "00_manifest"
        manifest_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "split": ["gallery", "query"],
                "index": [1, 1],
                "image_path": [
                    "/data/images/a.jpg",
                    "/data/images/b.jpg",
                ],
                "image_name": ["a.jpg", "b.jpg"],
            }
        ).to_csv(manifest_dir / "all_images.csv", index=False)
        (manifest_dir / "run_environment.json").write_text(
            json.dumps({"args": {"images_dir": run_images_dir}}),
            encoding="utf-8",
        )
        (manifest_dir / "split_params.json").write_text(
            json.dumps({"gallery_images_count": 1, "query_images_count": 1}),
            encoding="utf-8",
        )
        config = root / "config.yaml"
        config.write_text(
            "full_photo_identification:\n"
            f"  images_dir: '{config_images_dir}'\n",
            encoding="utf-8",
        )
        return exp, config

    def test_matching_source_paths_are_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exp, config = self._write_fixture(
                Path(tmp),
                "/data/images",
                "/data/images",
            )
            report = build_source_report(exp, config)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["manifest_images_count"], 2)

    def test_mismatching_config_and_run_are_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exp, config = self._write_fixture(
                Path(tmp),
                "/other/images",
                "/data/images",
            )
            report = build_source_report(exp, config)
            self.assertEqual(report["status"], "error")
            self.assertGreaterEqual(report["errors"], 1)


class ExportTests(unittest.TestCase):
    def test_export_works_with_minimal_experiment_and_sanitizes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "experiment"
            results_dir = exp / "04_identification"
            reports_dir = exp / "05_reports"
            results_dir.mkdir(parents=True)
            reports_dir.mkdir(parents=True)
            (results_dir / "identification_results.csv").write_text(
                "image_name,object_id,sku_status\nimage.jpg,1,unknown\n",
                encoding="utf-8",
            )
            (reports_dir / "full_experiment_summary.md").write_text(
                "# ShelfVision: отчёт для ВКР\n",
                encoding="utf-8",
            )
            output = exp / "export" / "demo_artifacts.zip"
            result = build_defense_export_zip(
                experiment_dir=exp,
                output_zip=output,
                include_visualizations=False,
            )
            self.assertTrue(result["zip"].exists())
            with zipfile.ZipFile(result["zip"]) as archive:
                names = set(archive.namelist())
                report_text = archive.read(
                    "experiment/05_reports/full_experiment_summary.md"
                ).decode("utf-8")
            self.assertIn(
                "experiment/04_identification/identification_results.csv",
                names,
            )
            self.assertIn("EXPORT_MANIFEST.json", names)
            self.assertNotIn("ShelfVision", report_text)
            self.assertNotIn("для ВКР", report_text)


if __name__ == "__main__":
    unittest.main()
