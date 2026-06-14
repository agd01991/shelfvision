from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml


KEY_PACKAGES = ["numpy", "pandas", "opencv-python", "ultralytics", "streamlit", "torch"]
STATUS_COLUMNS = ["status", "sku_status", "assignment_status"]
SIMILARITY_COLUMNS = ["best_similarity", "similarity", "score", "sku_confidence"]
MARGIN_COLUMNS = ["margin", "distinct_margin"]


@dataclass
class RunMetadataOutputs:
    run_config_yaml: str
    run_manifest_json: str
    environment_txt: str


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_existing(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    return None


def _status_column(df: pd.DataFrame) -> Optional[str]:
    for col in STATUS_COLUMNS:
        if col in df.columns:
            return col
    return None


def _mean_from_columns(df: pd.DataFrame, columns: Iterable[str]) -> float:
    for col in columns:
        if col in df.columns:
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())
    return 0.0


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except Exception:
        return "не найден"


def _cuda_status() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"доступна: {torch.cuda.get_device_name(0)}"
        return "недоступна"
    except Exception as exc:
        return f"не удалось проверить: {exc}"


def collect_run_counts(run_dir: Path) -> Dict[str, int | float | str]:
    summary_csv = _first_existing(run_dir, ["03_query_inference/summary.csv", "01_inference/summary.csv", "summary.csv"])
    crops_csv = _first_existing(
        run_dir,
        [
            "04_identification/crops_manifest.csv",
            "03_query_crops/crops_manifest.csv",
            "02_demo_gallery/crops_manifest.csv",
            "crops_manifest.csv",
        ],
    )
    gallery_csv = _first_existing(run_dir, ["02_demo_gallery/sku_gallery_final/gallery.csv", "02_demo_gallery/gallery.csv", "gallery.csv"])
    results_csv = _first_existing(run_dir, ["04_identification/identification_results.csv", "03_identification/identification_results.csv", "identification_results.csv"])

    summary_df = _read_csv(summary_csv)
    crops_df = _read_csv(crops_csv)
    gallery_df = _read_csv(gallery_csv)
    results_df = _read_csv(results_csv)

    detected_objects = 0
    if "objects_count" in summary_df.columns:
        detected_objects = int(pd.to_numeric(summary_df["objects_count"], errors="coerce").fillna(0).sum())
    elif not results_df.empty:
        detected_objects = len(results_df)

    gallery_items = int(gallery_df["sku_id"].nunique()) if "sku_id" in gallery_df.columns else 0
    gallery_refs = int(len(gallery_df)) if not gallery_df.empty else 0

    status_col = _status_column(results_df)
    if status_col:
        statuses = results_df[status_col].astype(str)
        matched = int(statuses.eq("matched").sum())
        matched_uncertain = int(statuses.eq("matched_uncertain").sum())
        unknown = int(statuses.eq("unknown").sum())
    else:
        matched = matched_uncertain = unknown = 0

    avg_similarity = _mean_from_columns(results_df, SIMILARITY_COLUMNS)
    avg_margin = _mean_from_columns(results_df, MARGIN_COLUMNS)

    return {
        "processed_images": int(len(summary_df)) if not summary_df.empty else 0,
        "detected_objects": int(detected_objects),
        "crops_count": int(len(crops_df)) if not crops_df.empty else 0,
        "gallery_items": gallery_items,
        "gallery_refs": gallery_refs,
        "identification_rows": int(len(results_df)) if not results_df.empty else 0,
        "status_column": status_col or "",
        "matched": matched,
        "matched_uncertain": matched_uncertain,
        "unknown": unknown,
        "avg_similarity": avg_similarity,
        "avg_margin": avg_margin,
    }


def collect_created_files(run_dir: Path, limit: int = 500) -> List[str]:
    if not run_dir.exists():
        return []
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    result = []
    for path in files[:limit]:
        try:
            result.append(str(path.relative_to(run_dir)).replace("\\", "/"))
        except Exception:
            result.append(str(path))
    return result


def write_environment(out_dir: Path) -> Path:
    path = out_dir / "environment.txt"
    lines = [
        f"Python version: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"Working directory: {Path.cwd()}",
        f"GPU/CUDA availability: {_cuda_status()}",
        "Installed key packages:",
    ]
    for package in KEY_PACKAGES:
        lines.append(f"- {package}: {_package_version(package)}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_run_metadata(run_dir: str | Path, params: Dict[str, Any]) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "run_config.yaml"
    manifest_path = run_dir / "run_manifest.json"
    environment_path = write_environment(run_dir)

    config_path.write_text(yaml.safe_dump(params, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest = {
        "run_datetime": datetime.now().isoformat(timespec="seconds"),
        "model": params.get("model", ""),
        "weights": params.get("weights", ""),
        "images_dir": params.get("images_dir", ""),
        "out_dir": str(run_dir),
        "conf": params.get("conf", None),
        "imgsz": params.get("imgsz", None),
        "threshold": params.get("threshold", None),
        "top_k": params.get("top_k", None),
        **collect_run_counts(run_dir),
        "created_files": collect_created_files(run_dir),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_config_yaml": config_path,
        "run_manifest_json": manifest_path,
        "environment_txt": environment_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сохранение паспорта запуска ShelfVision")
    parser.add_argument("--run-dir", required=True, help="Папка результата запуска")
    parser.add_argument("--model", default="", help="Модель")
    parser.add_argument("--weights", default="", help="Путь к весам")
    parser.add_argument("--images-dir", default="", help="Папка изображений")
    parser.add_argument("--conf", type=float, default=None, help="Порог уверенности")
    parser.add_argument("--imgsz", type=int, default=None, help="Размер изображения")
    parser.add_argument("--threshold", type=float, default=None, help="Порог SKU-сопоставления")
    parser.add_argument("--top-k", type=int, default=None, help="Количество ближайших SKU-кандидатов")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = {key: value for key, value in vars(args).items() if key != "run_dir" and value is not None}
    outputs = write_run_metadata(args.run_dir, params=params)
    print("=== ShelfVision: паспорт запуска сохранён ===")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
