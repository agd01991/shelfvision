from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd


def P(p: str | None) -> Optional[Path]:
    if not p:
        return None
    x = Path(p)
    return x if x.exists() else None


def read_json(p: Path) -> dict:
    """
    Читает JSON-файл.
    Если файл содержит несколько JSON подряд (или JSONL), пробует взять первый объект.
    """
    txt = p.read_text(encoding="utf-8").strip()

    # обычный JSON
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass

    # попробуем взять первый JSON-объект из начала файла
    decoder = json.JSONDecoder()
    try:
        obj, idx = decoder.raw_decode(txt)
        return obj if isinstance(obj, dict) else {"_parsed_first": obj}
    except Exception:
        return {"_error": "invalid_json", "_path": str(p)}


def load_summary_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # ожидаем твой формат: exp, P, R, mAP50, mAP50-95, seconds_total, status, model, override
    for c in ["P", "R", "mAP50", "mAP50-95", "seconds_total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "seconds_total" in df.columns:
        df["minutes_total"] = (df["seconds_total"] / 60.0).round(2)
    # пометим лучший запуск
    if "mAP50-95" in df.columns and len(df):
        rank = df["mAP50-95"].fillna(-1) * 1000 + df.get("mAP50", pd.Series([-1]*len(df))).fillna(-1)
        df["is_best"] = False
        df.loc[rank.idxmax(), "is_best"] = True
    return df


def load_dir3_metrics_json(p: Path) -> pd.DataFrame:
    d = read_json(p)
    mt = d.get("metrics_test", {})
    rows = []
    for k in ["yolo", "rtdetr", "wbf"]:
        if isinstance(mt.get(k), dict):
            row = {"system": k}
            row.update(mt[k])
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in ["AP", "AP50", "AP75", "AR1", "AR10", "AR100"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_dir5_robustness(dir_path: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(dir_path.glob("metrics_*.json")):
        d = read_json(p)
        m = d.get("metrics", {})
        rows.append({
            "mode": d.get("mode"),
            "param": d.get("param"),
            "AP": m.get("AP"),
            "AP50": m.get("AP50"),
            "AP75": m.get("AP75"),
            "AR100": m.get("AR100"),
            "file": str(p),
        })
    df = pd.DataFrame(rows)
    for c in ["param", "AP", "AP50", "AP75", "AR100"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_ultralytics_results_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    return df


def load_d2s_yolo_seg_run(run_dir: Path) -> pd.DataFrame:
    res = run_dir / "results.csv"
    if not res.exists():
        return pd.DataFrame([{"run_dir": str(run_dir), "status": "missing_results.csv"}])
    df = load_ultralytics_results_csv(res)
    last = df.iloc[-1].to_dict() if len(df) else {}
    # Ultralytics может писать разные ключи; берём самые частые
    def pick(keys: List[str]):
        for k in keys:
            if k in last and str(last[k]) != "" and not pd.isna(last[k]):
                try:
                    return float(last[k])
                except Exception:
                    return last[k]
        return None
    out = {
        "run_dir": str(run_dir),
        "status": "ok" if len(df) else "empty",
        "epoch_last": int(pick(["epoch"])) if pick(["epoch"]) is not None else None,
        "P_box": pick(["metrics/precision(B)", "metrics/precision"]),
        "R_box": pick(["metrics/recall(B)", "metrics/recall"]),
        "mAP50_box": pick(["metrics/mAP50(B)", "metrics/mAP50"]),
        "mAP5095_box": pick(["metrics/mAP50-95(B)", "metrics/mAP50-95"]),
        "mAP50_mask": pick(["metrics/mAP50(M)"]),
        "mAP5095_mask": pick(["metrics/mAP50-95(M)"]),
    }
    return pd.DataFrame([out])


def load_dir1_metrics_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_json", default="reports/path_scan/paths.json")
    ap.add_argument("--out_dir", default="reports/all_stats")
    args = ap.parse_args()

    paths = read_json(Path(args.paths_json).resolve())
    repo_root = Path(paths["ROOT_repo"]["path"]).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load what exists ---
    sheets: Dict[str, pd.DataFrame] = {}

    # 11 YOLO ablations
    p_summary = P(paths.get("YOLO_11_summary_csv", {}).get("path"))
    if p_summary:
        sheets["YOLO_11_ablations"] = load_summary_csv(p_summary)

    # dir1 models compare
    p_dir1 = P(paths.get("DIR1_metrics_csv", {}).get("path"))
    if p_dir1:
        sheets["DIR1_models"] = load_dir1_metrics_csv(p_dir1)
    else:
        sheets["DIR1_models_missing"] = pd.DataFrame([{
            "status": "missing",
            "note": "DIR1_metrics_csv not found. Re-run scripts/dir1_report_and_wbf.py to generate metrics.csv"
        }])

    # dir3 WBF
    p_dir3 = P(paths.get("DIR3_metrics_json", {}).get("path"))
    if p_dir3:
        sheets["DIR3_WBF"] = load_dir3_metrics_json(p_dir3)

    # dir5 robustness
    p_dir5 = P(paths.get("DIR5_robustness_dir", {}).get("path"))
    if p_dir5:
        sheets["DIR5_robustness"] = load_dir5_robustness(p_dir5)

    # D2S YOLO-seg run (latest)
    p_d2s_run = P(paths.get("D2S_yolo_seg_run_dir", {}).get("path"))
    if p_d2s_run:
        sheets["D2S_YOLO_SEG_last"] = load_d2s_yolo_seg_run(p_d2s_run)
    else:
        sheets["D2S_YOLO_SEG_missing"] = pd.DataFrame([{
            "status": "missing",
            "note": "D2S_yolo_seg_run_dir not found. It may be under ~/runs or you ran training from another cwd."
        }])

    # D2S Mask R-CNN metrics (whatever was found)
    p_d2s_mask_metrics = P(paths.get("D2S_maskrcnn_test_metrics", {}).get("path"))
    if p_d2s_mask_metrics:
        d = read_json(p_d2s_mask_metrics)
        row = {"path": str(p_d2s_mask_metrics), "status": "ok"}

        # detectron2 COCOEvaluator часто даёт {"bbox": {...}, "segm": {...}}
        if isinstance(d, dict) and "bbox" in d and isinstance(d["bbox"], dict):
            row["bbox_AP"] = d["bbox"].get("AP")
            row["bbox_AP50"] = d["bbox"].get("AP50")
            row["bbox_AR100"] = d["bbox"].get("AR100")
        if isinstance(d, dict) and "segm" in d and isinstance(d["segm"], dict):
            row["segm_AP"] = d["segm"].get("AP")
            row["segm_AP50"] = d["segm"].get("AP50")
            row["segm_AR100"] = d["segm"].get("AR100")

        # если не нашли ожидаемых ключей — просто сохраним список ключей
        if "bbox_AP" not in row and "segm_AP" not in row and isinstance(d, dict):
            row["raw_keys"] = ", ".join(list(d.keys())[:50])
        if isinstance(d, dict) and d.get("_error") == "invalid_json":
            row["status"] = "invalid_json"

        sheets["D2S_MaskRCNN_metrics_json"] = pd.DataFrame([row])

    # --- overall detection table (best YOLO + dir3 WBF if any) ---
    overall_rows = []
    if "YOLO_11_ablations" in sheets:
        df = sheets["YOLO_11_ablations"]
        if "is_best" in df.columns:
            b = df[df["is_best"] == True].iloc[0]
            overall_rows.append({
                "system": "YOLO_best_from_11",
                "name": b.get("exp"),
                "AP50-95": b.get("mAP50-95"),
                "AP50": b.get("mAP50"),
                "P": b.get("P"),
                "R": b.get("R"),
                "source": str(p_summary),
            })
    if "DIR3_WBF" in sheets:
        df = sheets["DIR3_WBF"]
        for _, r in df.iterrows():
            overall_rows.append({
                "system": f"DIR3_{r.get('system')}",
                "AP50-95": r.get("AP"),
                "AP50": r.get("AP50"),
                "AR100": r.get("AR100"),
                "source": str(p_dir3),
            })
    sheets["OVERALL_detection_min"] = pd.DataFrame(overall_rows)

    # --- save xlsx + csv copies ---
    xlsx_path = out_dir / "all_stats.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False)

    for name, df in sheets.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    print("DONE:", xlsx_path)
    print("OUT DIR:", out_dir)


if __name__ == "__main__":
    main()