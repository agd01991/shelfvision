from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Path helpers (WSL-friendly)
# -------------------------
def normalize_path_str(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    # Windows drive path -> /mnt/<drive>/...
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", p):
        drive = p[0].lower()
        tail = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{tail}"
    return p


def P(p: str | Path) -> Path:
    return Path(normalize_path_str(str(p))).expanduser().resolve()


def find_first(root: Path, rel_candidates: List[str]) -> Optional[Path]:
    for c in rel_candidates:
        x = root / c
        if x.exists():
            return x.resolve()
    return None


def find_glob(root: Path, pattern: str, limit: int = 20) -> List[Path]:
    xs = list(root.rglob(pattern))
    xs.sort()
    return xs[:limit]


# -------------------------
# Parsing / loading
# -------------------------
def load_summary_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names if needed
    # expected columns (typical): exp, mAP50, mAP50-95, P, R, seconds_total, status, model, override
    # Sometimes: "mAP50-95" could be "mAP50-95(B)" etc. Here it is summary from your runner, so likely stable.
    def pick_col(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_map = {
        "exp": pick_col(["exp", "run", "name"]),
        "mAP50": pick_col(["mAP50", "metrics/mAP50(B)", "metrics/mAP50"]),
        "mAP5095": pick_col(["mAP50-95", "mAP50-95(B)", "metrics/mAP50-95(B)", "metrics/mAP50-95"]),
        "P": pick_col(["P", "precision", "metrics/precision(B)", "metrics/precision"]),
        "R": pick_col(["R", "recall", "metrics/recall(B)", "metrics/recall"]),
        "seconds_total": pick_col(["seconds_total", "time_s", "seconds"]),
        "status": pick_col(["status"]),
        "model": pick_col(["model"]),
        "override": pick_col(["override"]),
    }

    # Keep only known columns + keep the rest as raw
    if col_map["exp"] is None:
        raise ValueError("summary.csv: cannot find experiment name column (exp/run/name).")

    df2 = pd.DataFrame()
    df2["exp"] = df[col_map["exp"]].astype(str)

    for k in ["mAP50", "mAP5095", "P", "R", "seconds_total", "status", "model", "override"]:
        c = col_map.get(k)
        if c and c in df.columns:
            df2[k] = df[c]
        else:
            df2[k] = None

    # Convert numerics
    for c in ["mAP50", "mAP5095", "P", "R", "seconds_total"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    df2["minutes_total"] = (df2["seconds_total"] / 60.0).round(2)

    # Parse ablation type/value from exp name
    def parse_variant(name: str) -> Tuple[str, str]:
        n = name
        if n.startswith("BASE"):
            return ("base", "")
        if "model_yolov8" in n:
            # E01_model_yolov8n
            m = re.search(r"model_(yolov8[nsmlx])", n)
            return ("model", m.group(1) if m else "")
        if "imgsz_" in n:
            m = re.search(r"imgsz_(\d+)", n)
            return ("imgsz", m.group(1) if m else "")
        if "lr0_" in n:
            m = re.search(r"lr0_(.+)", n)
            return ("lr0", (m.group(1) if m else "").replace("p", "."))
        if "wd_" in n:
            m = re.search(r"wd_(.+)", n)
            return ("weight_decay", (m.group(1) if m else "").replace("p", "."))
        if "optimizer_" in n:
            m = re.search(r"optimizer_(.+)", n)
            return ("optimizer", m.group(1) if m else "")
        if "mosaic_" in n:
            m = re.search(r"mosaic_(.+)", n)
            return ("mosaic", (m.group(1) if m else "").replace("p", "."))
        if "mixup_" in n:
            m = re.search(r"mixup_(.+)", n)
            return ("mixup", (m.group(1) if m else "").replace("p", "."))
        if "close_mosaic_" in n:
            m = re.search(r"close_mosaic_(\d+)", n)
            return ("close_mosaic", m.group(1) if m else "")
        return ("other", "")

    parsed = df2["exp"].apply(parse_variant)
    df2["variant"] = [v for v, _ in parsed]
    df2["variant_value"] = [x for _, x in parsed]

    # Best pick (by mAP50-95, then mAP50)
    df2["rank_key"] = df2["mAP5095"].fillna(-1) * 1000 + df2["mAP50"].fillna(-1)
    best_idx = df2["rank_key"].idxmax() if len(df2) else None
    df2["is_best"] = False
    if best_idx is not None and best_idx == best_idx:
        df2.loc[best_idx, "is_best"] = True

    return df2.drop(columns=["rank_key"])


def load_dir1_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns: model, AP50-95, AP50, AP_small, AP_medium, AP_large, ms_per_image
    return df


def load_dir3_metrics_json(path: Path) -> pd.DataFrame:
    d = json.loads(path.read_text(encoding="utf-8"))
    mt = d.get("metrics_test", {})
    rows = []
    for k in ["yolo", "rtdetr", "wbf"]:
        if k in mt:
            row = {"system": k}
            row.update(mt[k])
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def load_dir5_robustness_dir(dir_path: Path) -> pd.DataFrame:
    files = sorted(dir_path.glob("metrics_*.json"))
    rows = []
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        mode = d.get("mode")
        param = d.get("param")
        metrics = d.get("metrics", {})
        rows.append(
            {
                "mode": mode,
                "param": float(param) if param is not None else None,
                "AP": float(metrics.get("AP", float("nan"))),
                "AP50": float(metrics.get("AP50", float("nan"))),
                "AP75": float(metrics.get("AP75", float("nan"))),
                "AR100": float(metrics.get("AR100", float("nan"))),
                "file": str(p),
            }
        )
    df = pd.DataFrame(rows)
    return df


# -------------------------
# Plotting
# -------------------------
def save_plot_yolo_ablations(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart sorted by mAP50-95
    d = df.copy()
    d = d.sort_values(["mAP5095", "mAP50"], ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(d["exp"], d["mAP5095"])
    plt.xticks(rotation=30, ha="right")
    plt.title("YOLO ablations: mAP50-95 (sorted)")
    plt.tight_layout()
    plt.savefig(out_dir / "yolo_ablations_map5095.png", dpi=160)
    plt.close()

    # imgsz curve (if available)
    imgsz = d[d["variant"] == "imgsz"].copy()
    if len(imgsz):
        imgsz["imgsz"] = pd.to_numeric(imgsz["variant_value"], errors="coerce")
        imgsz = imgsz.dropna(subset=["imgsz"]).sort_values("imgsz")
        plt.figure(figsize=(6, 4))
        plt.plot(imgsz["imgsz"], imgsz["mAP5095"], marker="o")
        plt.title("YOLO: effect of imgsz on mAP50-95")
        plt.xlabel("imgsz")
        plt.ylabel("mAP50-95")
        plt.tight_layout()
        plt.savefig(out_dir / "yolo_imgsz_curve.png", dpi=160)
        plt.close()

    # model size bar (n/s/m)
    ms = d[d["variant"] == "model"].copy()
    if len(ms):
        plt.figure(figsize=(6, 4))
        plt.bar(ms["variant_value"], ms["mAP5095"])
        plt.title("YOLO: model size vs mAP50-95")
        plt.xlabel("model")
        plt.ylabel("mAP50-95")
        plt.tight_layout()
        plt.savefig(out_dir / "yolo_modelsize_bar.png", dpi=160)
        plt.close()


def save_plot_dir1_compare(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if "model" in df.columns and "AP50-95" in df.columns:
        d = df.copy()
        plt.figure(figsize=(8, 4))
        plt.bar(d["model"], d["AP50-95"])
        plt.xticks(rotation=20, ha="right")
        plt.title("Direction 1: AP50-95 (COCO) on TEST")
        plt.tight_layout()
        plt.savefig(out_dir / "dir1_ap5095_bar.png", dpi=160)
        plt.close()

    if "ms_per_image" in df.columns and "AP50-95" in df.columns:
        d = df.copy()
        plt.figure(figsize=(6, 4))
        plt.scatter(d["ms_per_image"], d["AP50-95"])
        for _, r in d.iterrows():
            plt.annotate(str(r["model"]), (r["ms_per_image"], r["AP50-95"]))
        plt.title("Direction 1: quality vs speed")
        plt.xlabel("ms/image")
        plt.ylabel("AP50-95")
        plt.tight_layout()
        plt.savefig(out_dir / "dir1_quality_vs_speed.png", dpi=160)
        plt.close()


def save_plot_dir5_robustness(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    # Baseline clean (param=0)
    clean = df[df["mode"] == "clean"].copy()
    ap_clean = None
    if len(clean):
        # take best/first clean
        ap_clean = float(clean.iloc[0]["AP"])

    # Plot each mode separately (clear and easy for WKR)
    for mode in sorted(df["mode"].dropna().unique()):
        if mode == "clean":
            continue
        dm = df[df["mode"] == mode].copy()
        if dm.empty:
            continue
        dm = dm.sort_values("param")
        plt.figure(figsize=(6, 4))
        plt.plot(dm["param"], dm["AP"], marker="o")
        plt.title(f"Robustness: {mode} (AP50-95)")
        plt.xlabel("param")
        plt.ylabel("AP50-95")
        if ap_clean is not None and not math.isnan(ap_clean):
            plt.axhline(ap_clean, linestyle="--")
        plt.tight_layout()
        plt.savefig(out_dir / f"robust_{mode}_ap.png", dpi=160)
        plt.close()


# -------------------------
# Thesis generator (for WKR)
# -------------------------
def make_theses(
    yolo_df: pd.DataFrame,
    dir1_df: Optional[pd.DataFrame],
    dir3_df: Optional[pd.DataFrame],
    dir5_df: Optional[pd.DataFrame],
) -> str:
    lines = []
    lines.append("# Итоговые выводы по экспериментам\n")

    # YOLO ablations
    if not yolo_df.empty:
        best = yolo_df[yolo_df["is_best"] == True]
        if len(best):
            b = best.iloc[0]
            lines.append("## YOLO: абляционные эксперименты\n")
            lines.append(
                f"- Лучший конфиг среди 11 прогонов: **{b['exp']}** "
                f"(mAP50-95={b['mAP5095']:.4f}, mAP50={b['mAP50']:.4f}, время≈{b['minutes_total']:.1f} мин)."
            )
            # imgsz note
            imgsz = yolo_df[yolo_df["variant"] == "imgsz"].dropna(subset=["mAP5095"])
            if len(imgsz) >= 2:
                best_imgsz = imgsz.sort_values("mAP5095", ascending=False).iloc[0]
                worst_imgsz = imgsz.sort_values("mAP5095", ascending=True).iloc[0]
                lines.append(
                    f"- Влияние размера входа: лучший результат при **imgsz={best_imgsz['variant_value']}**, "
                    f"худший при **imgsz={worst_imgsz['variant_value']}** (по mAP50-95)."
                )
            # model size note
            ms = yolo_df[yolo_df["variant"] == "model"].dropna(subset=["mAP5095"])
            if len(ms) >= 2:
                ms_sorted = ms.sort_values("mAP5095", ascending=False)
                lines.append(
                    f"- Сравнение размеров модели: лучший из (n/s/m) — **{ms_sorted.iloc[0]['variant_value']}**, "
                    f"самый слабый — **{ms_sorted.iloc[-1]['variant_value']}** (по mAP50-95)."
                )

    # Direction 1
    if dir1_df is not None and not dir1_df.empty:
        lines.append("\n## Направление 1: сравнение классов детекторов\n")
        # sort by AP
        if "AP50-95" in dir1_df.columns and "model" in dir1_df.columns:
            d = dir1_df.sort_values("AP50-95", ascending=False)
            top = d.iloc[0]
            lines.append(
                f"- На тестовом наборе лучший результат по AP50-95 показала система: **{top['model']}** "
                f"(AP50-95={top['AP50-95']:.4f}, AP50={top.get('AP50', float('nan')):.4f})."
            )
            if "ms_per_image" in d.columns:
                fastest = d.sort_values("ms_per_image", ascending=True).iloc[0]
                lines.append(
                    f"- Самая быстрая по скорости (мс/кадр): **{fastest['model']}** "
                    f"(≈{fastest['ms_per_image']:.1f} ms/img)."
                )

    # Direction 3 WBF
    if dir3_df is not None and not dir3_df.empty:
        lines.append("\n## Направление 3: ансамбль (WBF)\n")
        if set(["system", "AP"]).issubset(dir3_df.columns):
            # systems: yolo, rtdetr, wbf
            def ap(sysname: str) -> Optional[float]:
                s = dir3_df[dir3_df["system"] == sysname]
                if len(s):
                    return float(s.iloc[0]["AP"])
                return None

            ap_y = ap("yolo")
            ap_d = ap("rtdetr")
            ap_w = ap("wbf")

            if ap_w is not None:
                lines.append(f"- Ансамбль **WBF(YOLO+RT-DETR)** показал AP50-95={ap_w:.4f} на тесте.")
            if ap_y is not None and ap_w is not None:
                lines.append(f"- Прирост относительно YOLO: ΔAP={ap_w - ap_y:+.4f}.")
            if ap_d is not None and ap_w is not None:
                lines.append(f"- Прирост относительно RT-DETR: ΔAP={ap_w - ap_d:+.4f}.")

    # Direction 5 robustness
    if dir5_df is not None and not dir5_df.empty:
        lines.append("\n## Направление 5: устойчивость (robustness)\n")
        clean = dir5_df[dir5_df["mode"] == "clean"]
        ap_clean = float(clean.iloc[0]["AP"]) if len(clean) else None
        if ap_clean is not None:
            lines.append(f"- Базовое качество без искажений: AP50-95={ap_clean:.4f}.")
        # worst mode
        non_clean = dir5_df[dir5_df["mode"] != "clean"].dropna(subset=["AP"])
        if len(non_clean):
            worst = non_clean.sort_values("AP", ascending=True).iloc[0]
            lines.append(
                f"- Наиболее критичное ухудшение в текущем наборе тестов: **{worst['mode']}** "
                f"(param={worst['param']}, AP50-95={worst['AP']:.4f})."
            )
            best = non_clean.sort_values("AP", ascending=False).iloc[0]
            lines.append(
                f"- Наиболее мягкое ухудшение: **{best['mode']}** "
                f"(param={best['param']}, AP50-95={best['AP']:.4f})."
            )

    lines.append("\n---\n")
    lines.append("### Что считается практическим результатом\n")
    lines.append(
        "- Собран воспроизводимый пайплайн экспериментов: подготовка данных, обучение, оценка, агрегация результатов.\n"
        "- Сформирована рекомендация по выбору конфигурации под ограничения по времени/ресурсам.\n"
        "- Проверена гипотеза о комплементарности ошибок разных классов моделей через ансамбль WBF.\n"
        "- Оценена устойчивость к типовым ухудшениям качества изображения.\n"
    )

    return "\n".join(lines)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", help="repo root (for auto-search)")
    ap.add_argument("--summary_csv", default="", help="path to summary.csv (YOLO ablations)")
    ap.add_argument("--dir1_metrics_csv", default="", help="path to dir1 metrics.csv (from dir1_report_and_wbf)")
    ap.add_argument("--dir3_metrics_json", default="", help="path to dir3 wbf metrics.json")
    ap.add_argument("--dir5_dir", default="", help="path to dir5 robustness folder with metrics_*.json")
    ap.add_argument("--out_dir", default="reports/master", help="output folder")

    args = ap.parse_args()

    root = P(args.project_root)
    out_dir = P(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- locate files (auto) if not provided ---
    summary_csv = P(args.summary_csv) if args.summary_csv else None
    if summary_csv is None or not summary_csv.exists():
        # typical places
        cand = find_first(root, ["runs_night_backup/summary.csv", "summary.csv"])
        if cand is None and Path("/mnt/d/1Diplom").exists():
            cand = find_first(Path("/mnt/d/1Diplom"), ["summary.csv", "runs_night_backup/summary.csv"])
        summary_csv = cand

    dir1_metrics_csv = P(args.dir1_metrics_csv) if args.dir1_metrics_csv else None
    if dir1_metrics_csv is None or not dir1_metrics_csv.exists():
        # try find metrics.csv under dir1_compare
        matches = find_glob(root, "metrics.csv", limit=50)
        # pick likely one
        pick = None
        for m in matches:
            if "dir1" in str(m).lower() and "compare" in str(m).lower():
                pick = m
                break
        dir1_metrics_csv = pick

    dir3_metrics_json = P(args.dir3_metrics_json) if args.dir3_metrics_json else None
    if dir3_metrics_json is None or not dir3_metrics_json.exists():
        cand = find_first(root, ["artifacts/dir3_wbf/metrics.json", "artifacts/dir3_wbf_debug/metrics.json"])
        if cand is None and Path("/mnt/d/1Diplom").exists():
            cand = find_first(Path("/mnt/d/1Diplom"), ["reports/dir3_wbf/metrics.json", "artifacts/dir3_wbf/metrics.json"])
        dir3_metrics_json = cand

    dir5_dir = P(args.dir5_dir) if args.dir5_dir else None
    if dir5_dir is None or not dir5_dir.exists():
        cand = find_first(root, ["artifacts/dir5_robustness", "artifacts/dir5"])
        if cand is None and Path("/mnt/d/1Diplom").exists():
            cand = find_first(Path("/mnt/d/1Diplom"), ["artifacts/dir5_robustness", "reports/dir5_robustness"])
        dir5_dir = cand

    # --- load data ---
    if summary_csv is None or not summary_csv.exists():
        raise FileNotFoundError("summary.csv not found. Pass --summary_csv explicitly.")
    yolo_df = load_summary_csv(summary_csv)

    dir1_df = None
    if dir1_metrics_csv is not None and dir1_metrics_csv.exists():
        dir1_df = load_dir1_metrics_csv(dir1_metrics_csv)

    dir3_df = None
    if dir3_metrics_json is not None and dir3_metrics_json.exists():
        dir3_df = load_dir3_metrics_json(dir3_metrics_json)

    dir5_df = None
    if dir5_dir is not None and dir5_dir.exists():
        dir5_df = load_dir5_robustness_dir(dir5_dir)

    # --- save CSV copies ---
    yolo_df.to_csv(out_dir / "yolo_ablations.csv", index=False)
    if dir1_df is not None:
        dir1_df.to_csv(out_dir / "dir1_models.csv", index=False)
    if dir3_df is not None:
        dir3_df.to_csv(out_dir / "dir3_wbf.csv", index=False)
    if dir5_df is not None:
        dir5_df.to_csv(out_dir / "dir5_robustness.csv", index=False)

    # --- excel report ---
    xlsx_path = out_dir / "report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        yolo_df.to_excel(w, sheet_name="YOLO_ablations", index=False)
        if dir1_df is not None:
            dir1_df.to_excel(w, sheet_name="Dir1_models", index=False)
        if dir3_df is not None:
            dir3_df.to_excel(w, sheet_name="Dir3_WBF", index=False)
        if dir5_df is not None:
            dir5_df.to_excel(w, sheet_name="Dir5_robustness", index=False)

    # --- plots ---
    plots_dir = out_dir / "plots"
    save_plot_yolo_ablations(yolo_df, plots_dir)
    if dir1_df is not None:
        save_plot_dir1_compare(dir1_df, plots_dir)
    if dir5_df is not None:
        save_plot_dir5_robustness(dir5_df, plots_dir)

    # --- theses ---
    theses = make_theses(yolo_df, dir1_df, dir3_df, dir5_df)
    (out_dir / "theses.md").write_text(theses, encoding="utf-8")

    # --- console summary ---
    print("=== MASTER REPORT READY ===")
    print("summary_csv:", summary_csv)
    print("dir1_metrics_csv:", dir1_metrics_csv)
    print("dir3_metrics_json:", dir3_metrics_json)
    print("dir5_dir:", dir5_dir)
    print("out_dir:", out_dir)
    print("xlsx:", xlsx_path)
    print("theses:", out_dir / "theses.md")
    print("plots:", plots_dir)


if __name__ == "__main__":
    main()