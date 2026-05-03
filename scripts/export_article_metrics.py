from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ARTICLE1_TILING_EXPECTED = "reports/article_eval/article1_tiling_robustness.csv"
ARTICLE2_MODEL_ROBUSTNESS_EXPECTED = "reports/article_eval/article2_model_robustness.csv"

MODEL_PARADIGM = {
    "yolov8s": "one-stage",
    "yolo": "one-stage",
    "rt-detr-l": "transformer-based",
    "rtdetr-l": "transformer-based",
    "rt-detr": "transformer-based",
    "rtdetr": "transformer-based",
    "faster r-cnn": "two-stage",
    "faster r-cnn (d2)": "two-stage",
    "wbf(yolo+rtdetr)": "ensemble",
    "wbf": "ensemble",
}

ARTICLE2_ORDER = ["YOLOv8s", "RT-DETR-L", "Faster R-CNN (D2)", "WBF(YOLO+RTDETR)"]


def read_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"_value": value}
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value, _ = decoder.raw_decode(text)
        return value if isinstance(value, dict) else {"_value": value}


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": str(path)}
    try:
        return read_json(path)
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc), "_path": str(path)}


def path_from(paths: dict[str, Any], key: str) -> Optional[Path]:
    value = paths.get(key, {})
    if isinstance(value, dict) and value.get("path"):
        return Path(value["path"])
    return None


def existing(path: Optional[Path]) -> Optional[Path]:
    return path if path and path.exists() else None


def add_status(df: pd.DataFrame, value: str = "ok") -> pd.DataFrame:
    """Добавляет/обновляет колонку status без падения, если она уже есть в CSV."""
    if "status" in df.columns:
        df["status"] = df["status"].fillna(value)
        df.loc[df["status"].astype(str).str.strip().eq(""), "status"] = value
    else:
        df.insert(0, "status", value)
    return df


def count_split_items(value: Any) -> Optional[int]:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("images", "image_ids", "ids", "items"):
            if key in value and isinstance(value[key], list):
                return len(value[key])
    return None


def pick_first(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def summarize_dataset(dataset_dir: Optional[Path], label: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_mode": label,
        "path": str(dataset_dir) if dataset_dir else "",
        "status": "missing",
        "images": None,
        "objects_or_annotations": None,
        "categories": None,
        "train": None,
        "val": None,
        "test": None,
        "tile_count": None,
        "tile_size": None,
        "stride": None,
        "overlap": None,
        "notes": "",
    }
    if not dataset_dir or not dataset_dir.exists():
        return row

    row["status"] = "found"

    passport = safe_read_json(dataset_dir / "passport.json")
    if not passport.get("_missing") and not passport.get("_error"):
        row["images"] = pick_first(passport, ["images", "num_images", "n_images", "images_count", "image_count"])
        row["objects_or_annotations"] = pick_first(
            passport,
            ["objects", "num_objects", "n_objects", "annotations", "num_annotations", "annotations_count", "bbox_count"],
        )
        row["categories"] = pick_first(passport, ["categories", "num_categories", "n_categories", "classes", "num_classes"])
        row["tile_size"] = pick_first(passport, ["tile_size", "tile", "tile_wh"])
        row["stride"] = pick_first(passport, ["stride", "tile_stride", "step"])
        row["overlap"] = pick_first(passport, ["overlap", "tile_overlap"])
        row["notes"] += "passport.json прочитан; "

    for name in ("annotations.json", "instances.json"):
        p = dataset_dir / name
        if p.exists():
            coco = safe_read_json(p)
            if isinstance(coco.get("images"), list):
                row["images"] = row["images"] or len(coco["images"])
            if isinstance(coco.get("annotations"), list):
                row["objects_or_annotations"] = row["objects_or_annotations"] or len(coco["annotations"])
            if isinstance(coco.get("categories"), list):
                row["categories"] = row["categories"] or len(coco["categories"])
            row["notes"] += f"{name} прочитан; "
            break

    for name in ("splits.json", "split.json"):
        p = dataset_dir / name
        if p.exists():
            splits = safe_read_json(p)
            for split_key in ("train", "val", "test"):
                row[split_key] = count_split_items(splits.get(split_key))
            row["notes"] += f"{name} прочитан; "
            break

    tile_map_path = dataset_dir / "tile_map.json"
    if tile_map_path.exists():
        tile_map = safe_read_json(tile_map_path)
        if isinstance(tile_map, dict):
            if isinstance(tile_map.get("tiles"), list):
                row["tile_count"] = len(tile_map["tiles"])
            elif isinstance(tile_map.get("tile_map"), list):
                row["tile_count"] = len(tile_map["tile_map"])
            else:
                row["tile_count"] = len(tile_map)
        row["notes"] += "tile_map.json прочитан; "

    return row


def load_dir1_models(paths: dict[str, Any]) -> pd.DataFrame:
    p = existing(path_from(paths, "DIR1_metrics_csv"))
    if not p:
        return pd.DataFrame([{"status": "missing", "note": "DIR1_metrics_csv не найден."}])
    df = pd.read_csv(p)
    add_status(df)
    df["source_file"] = str(p)
    if "model" in df.columns:
        df["paradigm"] = df["model"].map(lambda x: MODEL_PARADIGM.get(str(x).lower(), "уточнить"))
        order_map = {name: i for i, name in enumerate(ARTICLE2_ORDER)}
        df["_order"] = df["model"].map(lambda x: order_map.get(str(x), 999))
        df = df.sort_values("_order").drop(columns=["_order"])
    return df


def load_yolo_ablations(paths: dict[str, Any]) -> pd.DataFrame:
    p = existing(path_from(paths, "YOLO_11_summary_csv"))
    if not p:
        return pd.DataFrame([{"status": "missing", "note": "YOLO_11_summary_csv не найден."}])
    df = pd.read_csv(p)
    for col in ("P", "R", "mAP50", "mAP50-95", "seconds_total"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "seconds_total" in df.columns:
        df["minutes_total"] = (df["seconds_total"] / 60).round(2)
    if "mAP50-95" in df.columns and len(df):
        rank = df["mAP50-95"].fillna(-1) * 1000 + df.get("mAP50", pd.Series([-1] * len(df))).fillna(-1)
        df["is_best"] = False
        df.loc[rank.idxmax(), "is_best"] = True
    add_status(df)
    df["source_file"] = str(p)
    return df


def load_robustness_current(paths: dict[str, Any]) -> pd.DataFrame:
    robustness_dir = existing(path_from(paths, "DIR5_robustness_dir"))
    if not robustness_dir:
        return pd.DataFrame([{"status": "missing", "note": "DIR5_robustness_dir не найден."}])

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(robustness_dir.glob("metrics_*.json")):
        data = safe_read_json(metrics_path)
        metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
        rows.append(
            {
                "model": data.get("model", "current_base_detector"),
                "dataset_mode": data.get("dataset_mode", "current_or_unspecified"),
                "corruption": data.get("mode", metrics_path.stem.replace("metrics_", "")),
                "param": data.get("param"),
                "AP50-95": metrics.get("AP"),
                "AP50": metrics.get("AP50"),
                "AP75": metrics.get("AP75"),
                "AR100": metrics.get("AR100"),
                "source_file": str(metrics_path),
            }
        )

    if not rows:
        return pd.DataFrame([{"status": "missing", "note": f"В {robustness_dir} нет metrics_*.json."}])

    df = pd.DataFrame(rows)
    for col in ("param", "AP50-95", "AP50", "AP75", "AR100"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    clean_rows = df[df["corruption"].astype(str).str.lower().eq("clean")]
    if len(clean_rows):
        clean = clean_rows.iloc[0]
        for metric in ("AP50-95", "AP50", "AP75", "AR100"):
            df[f"delta_{metric}"] = (clean.get(metric) - df[metric]).round(6)
    else:
        for metric in ("AP50-95", "AP50", "AP75", "AR100"):
            df[f"delta_{metric}"] = None

    add_status(df)
    return df


def load_optional_extended_csv(repo_root: Path, relative_path: str, required_columns: set[str]) -> pd.DataFrame:
    p = repo_root / relative_path
    if not p.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "expected_file": str(p),
                    "note": "Опциональный файл не найден. Создай его, если хочешь закрыть все таблицы статей без ручной склейки.",
                }
            ]
        )
    df = pd.read_csv(p)
    missing = required_columns - set(df.columns)
    add_status(df, "ok" if not missing else "bad_columns")
    if missing:
        df["missing_columns"] = ", ".join(sorted(missing))
    df["source_file"] = str(p)
    return df


def make_article2_clean_table(dir1_models: pd.DataFrame) -> pd.DataFrame:
    if "model" not in dir1_models.columns:
        return dir1_models
    columns = [
        c
        for c in ["model", "paradigm", "AP50-95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "ms_per_image"]
        if c in dir1_models.columns
    ]
    df = dir1_models[columns].copy()
    for col in ("AP50-95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "ms_per_image"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(5)
    return df


def has_ok(df: pd.DataFrame) -> bool:
    return "status" in df.columns and (df["status"] == "ok").any()


def make_missing_checklist(
    dataset_summary: pd.DataFrame,
    article1_extended: pd.DataFrame,
    article2_extended: pd.DataFrame,
    robustness_current: pd.DataFrame,
    dir1_models: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article": "1",
                "item": "Характеристики non-tiled/tiled датасетов",
                "status": "ok" if len(dataset_summary) and (dataset_summary["status"] == "found").any() else "missing",
                "what_to_do": "Проверь dataset_summary.csv; если counts пустые, открой passport.json/annotations.json вручную.",
            },
            {
                "article": "1",
                "item": "Парное сравнение tiled vs non-tiled на clean/degraded",
                "status": "ok" if has_ok(article1_extended) else "missing",
                "what_to_do": f"Нужен файл {ARTICLE1_TILING_EXPECTED} с колонками model,dataset_mode,corruption,AP50-95,AP50,AP75,AR100,ms_per_image.",
            },
            {
                "article": "1/2",
                "item": "Текущая robustness-таблица по деградациям",
                "status": "ok" if has_ok(robustness_current) else "missing",
                "what_to_do": "Используй current_robustness_with_delta.csv как временную таблицу; она не заменяет tiled/non-tiled и 3-model robustness.",
            },
            {
                "article": "2",
                "item": "Clean-сравнение YOLO / RT-DETR / Faster R-CNN",
                "status": "ok" if "model" in dir1_models.columns else "missing",
                "what_to_do": "Используй article2_clean_models.csv.",
            },
            {
                "article": "2",
                "item": "Robustness YOLO / RT-DETR / Faster R-CNN",
                "status": "ok" if has_ok(article2_extended) else "missing",
                "what_to_do": f"Нужен файл {ARTICLE2_MODEL_ROBUSTNESS_EXPECTED} с колонками model,corruption,AP50-95,AP50,AP75,AR100,delta_AP50-95.",
            },
            {
                "article": "2",
                "item": "AR100 в clean-сравнении моделей",
                "status": "ok" if "AR100" in dir1_models.columns else "missing",
                "what_to_do": "Если AR100 нужен в статье 2, дополни DIR1_metrics_csv или отдельный article2_model_robustness.csv clean-строками.",
            },
        ]
    )


def write_markdown(out_dir: Path, sheets: dict[str, pd.DataFrame]) -> None:
    lines = ["# Таблицы для статей\n", "Сгенерировано скриптом `scripts/export_article_metrics.py`.\n"]
    for title, key in [
        ("Статья 2. Clean-сравнение моделей", "article2_clean_models"),
        ("Текущая robustness-таблица с Δ", "current_robustness_with_delta"),
        ("Чек-лист недостающих данных", "missing_checklist"),
    ]:
        df = sheets.get(key)
        if df is not None and len(df):
            lines.append(f"\n## {title}\n")
            try:
                lines.append(df.to_markdown(index=False))
            except Exception:  # noqa: BLE001
                lines.append(df.to_csv(index=False))
            lines.append("\n")
    (out_dir / "article_tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Собирает таблицы с метриками для двух статей по ВКР.")
    parser.add_argument("--paths_json", default="reports/path_scan/paths.json", help="Путь к paths.json")
    parser.add_argument("--out_dir", default="reports/article_metrics", help="Куда сохранить xlsx/csv/md")
    args = parser.parse_args()

    paths_json = Path(args.paths_json).resolve()
    paths = read_json(paths_json)
    repo_root = path_from(paths, "ROOT_repo") or Path.cwd()
    if not repo_root.exists():
        repo_root = Path.cwd()

    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = pd.DataFrame(
        [
            summarize_dataset(path_from(paths, "SKU_prepared_small_v1"), "non_tiled"),
            summarize_dataset(path_from(paths, "SKU_prepared_small_v1_tiled"), "tiled"),
        ]
    )
    dir1_models = load_dir1_models(paths)
    yolo_ablations = load_yolo_ablations(paths)
    robustness_current = load_robustness_current(paths)
    article1_extended = load_optional_extended_csv(
        repo_root,
        ARTICLE1_TILING_EXPECTED,
        {"model", "dataset_mode", "corruption", "AP50-95", "AP50", "AP75", "AR100"},
    )
    article2_extended = load_optional_extended_csv(
        repo_root,
        ARTICLE2_MODEL_ROBUSTNESS_EXPECTED,
        {"model", "corruption", "AP50-95", "AP50", "AP75", "AR100"},
    )

    sheets = {
        "dataset_summary": dataset_summary,
        "article2_clean_models": make_article2_clean_table(dir1_models),
        "current_robustness_with_delta": robustness_current,
        "article1_tiling_robustness": article1_extended,
        "article2_model_robustness": article2_extended,
        "dir1_models_raw": dir1_models,
        "yolo_ablations_raw": yolo_ablations,
    }
    sheets["missing_checklist"] = make_missing_checklist(
        dataset_summary, article1_extended, article2_extended, robustness_current, dir1_models
    )

    xlsx_path = out_dir / "article_metrics.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    for name, df in sheets.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    write_markdown(out_dir, sheets)

    print("DONE")
    print(f"XLSX: {xlsx_path}")
    print(f"CSV/MD dir: {out_dir}")
    print("Главные файлы:")
    print(f"  - {out_dir / 'article_metrics.xlsx'}")
    print(f"  - {out_dir / 'article_tables.md'}")
    print(f"  - {out_dir / 'missing_checklist.csv'}")


if __name__ == "__main__":
    main()
