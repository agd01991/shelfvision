from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -------------------------
# path helpers (WSL-friendly)
# -------------------------
def normalize_any_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    # Windows drive path -> /mnt/<drive>/...
    if re.match(r"^[A-Za-z]:[\\/]", p):
        try:
            return subprocess.check_output(["wslpath", "-u", p]).decode().strip()
        except Exception:
            drive = p[0].lower()
            tail = p[2:].lstrip("\\/").replace("\\", "/")
            return f"/mnt/{drive}/{tail}"
    return p


def P(p: str | Path) -> Path:
    return Path(normalize_any_path(str(p))).expanduser().resolve()


def first_existing(cands: List[Path]) -> Optional[Path]:
    for p in cands:
        if p and p.exists():
            return p
    return None


def rglob_best(root: Path, pattern: str, prefer_keywords: List[str] | None = None) -> Optional[Path]:
    """Find first match for pattern, optionally prefer paths containing keywords."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        return None
    if prefer_keywords:
        scored = []
        for m in matches:
            s = 0
            low = str(m).lower()
            for kw in prefer_keywords:
                if kw.lower() in low:
                    s += 1
            scored.append((s, m))
        scored.sort(key=lambda x: (-x[0], len(str(x[1]))))
        return scored[0][1]
    return matches[0]


def find_dir(root: Path, rel: str) -> Optional[Path]:
    p = root / rel
    return p if p.exists() else None


@dataclass
class FoundItem:
    key: str
    path: Optional[str]
    status: str
    note: str = ""


def add(items: List[FoundItem], key: str, p: Optional[Path], note: str = "") -> None:
    items.append(
        FoundItem(
            key=key,
            path=str(p) if p else None,
            status="FOUND" if p and p.exists() else "MISSING",
            note=note,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo_root",
        default=r"C:\Users\agd01\Documents\1ДипломМага\Проги\shelfvision",
        help="Path to shelfvision repo (Windows or WSL path)",
    )
    ap.add_argument(
        "--diplom_root",
        default=r"D:\1Diplom",
        help="Path to diplom artifacts root (Windows or WSL path)",
    )
    ap.add_argument("--out_dir", default="reports/path_scan", help="output folder (inside repo)")
    args = ap.parse_args()

    repo = P(args.repo_root)
    diplom = P(args.diplom_root)
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items: List[FoundItem] = []

    # --- roots ---
    add(items, "ROOT_repo", repo, "Корень репозитория shelfvision")
    add(items, "ROOT_diplom", diplom, "Папка D:\\1Diplom")

    # --- key datasets / prepared ---
    add(items, "SKU_prepared_small_v1", find_dir(repo, "data/prepared/sku110k_small/v1"), "Prepared COCO bbox (small)")
    add(items, "SKU_prepared_small_v1_tiled", find_dir(repo, "data/prepared/sku110k_small/v1_tiled"), "Prepared tiled")
    add(items, "SKU_yolo_cache_small_tiled", find_dir(repo, "data/yolo_cache/sku110k_small_v1_tiled"), "Ultralytics dataset cache")

    add(items, "D2S_raw_small", find_dir(repo, "data/raw/d2s_small"), "D2S small subset (images + annotations.json)")
    add(items, "D2S_splits", find_dir(repo, "data/coco_splits/d2s_small"), "train/val/test json (fix versions)")
    add(items, "D2S_yolo_seg_cache", find_dir(repo, "data/yolo_cache/d2s_small_seg"), "Ultralytics seg dataset cache")

    # --- important metrics/outputs ---
    # 11 YOLO ablations summary
    summary_csv = first_existing(
        [
            diplom / "runs_night_backup/summary.csv",
            repo / "runs_night_backup/summary.csv",
            repo / "summary.csv",
        ]
    )
    add(items, "YOLO_11_summary_csv", summary_csv, "Сводка 11 экспериментов YOLO (summary.csv)")

    # dir1 compare metrics.csv (quality vs speed)
    dir1_metrics = first_existing(
        [
            diplom / "reports/dir1_compare/metrics.csv",
            repo / "reports/dir1_compare/metrics.csv",
            rglob_best(diplom, "metrics.csv", prefer_keywords=["dir1", "compare"]) if diplom.exists() else None,
            rglob_best(repo, "metrics.csv", prefer_keywords=["dir1", "compare"]),
        ]
    )
    add(items, "DIR1_metrics_csv", dir1_metrics, "Таблица сравнения моделей (dir1_report_and_wbf)")

    # dir3 WBF metrics.json
    dir3_metrics = first_existing(
        [
            repo / "artifacts/dir3_wbf/metrics.json",
            diplom / "artifacts/dir3_wbf/metrics.json",
            rglob_best(repo, "metrics.json", prefer_keywords=["dir3", "wbf"]),
            rglob_best(diplom, "metrics.json", prefer_keywords=["dir3", "wbf"]) if diplom.exists() else None,
        ]
    )
    add(items, "DIR3_metrics_json", dir3_metrics, "WBF ensemble metrics.json")

    # dir5 robustness folder
    dir5_dir = first_existing(
        [
            repo / "artifacts/dir5_robustness",
            diplom / "artifacts/dir5_robustness",
            rglob_best(repo, "metrics_*.json", prefer_keywords=["dir5", "robustness"]).parent if rglob_best(repo, "metrics_*.json", prefer_keywords=["dir5", "robustness"]) else None,
            rglob_best(diplom, "metrics_*.json", prefer_keywords=["dir5", "robustness"]).parent if diplom.exists() and rglob_best(diplom, "metrics_*.json", prefer_keywords=["dir5", "robustness"]) else None,
        ]
    )
    add(items, "DIR5_robustness_dir", dir5_dir, "Папка с metrics_*.json по устойчивости")

    # D2S Mask R-CNN (weights + test metrics)
    d2s_mask_weights = first_existing(
        [
            repo / "runs/d2s_maskrcnn/model_final.pth",
            rglob_best(repo, "model_final.pth", prefer_keywords=["d2s", "maskrcnn"]),
        ]
    )
    add(items, "D2S_maskrcnn_model_final", d2s_mask_weights, "Detectron2 Mask R-CNN weights")

    d2s_mask_test_metrics = first_existing(
        [
            repo / "artifacts/d2s_maskrcnn_test/metrics.json",
            rglob_best(repo, "metrics.json", prefer_keywords=["d2s", "maskrcnn", "test"]),
        ]
    )
    add(items, "D2S_maskrcnn_test_metrics", d2s_mask_test_metrics, "Mask R-CNN evaluation on test_fix.json")

    # D2S YOLO-seg runs: find latest run folder by name pattern
    d2s_seg_runs = repo / "runs/d2s_seg"
    best_seg_run = None
    if d2s_seg_runs.exists():
        # prefer your run name with "img640"
        candidates = sorted([p for p in d2s_seg_runs.iterdir() if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        for p in candidates:
            if "d2s_small_yolov8s_seg_img640" in p.name:
                best_seg_run = p
                break
        if best_seg_run is None and candidates:
            best_seg_run = candidates[0]
    add(items, "D2S_yolo_seg_run_dir", best_seg_run, "Последний запуск YOLO-seg на D2S (папка run)")
    add(items, "D2S_yolo_seg_best_pt", (best_seg_run / "weights/best.pt") if best_seg_run else None, "YOLO-seg best.pt")
    add(items, "D2S_yolo_seg_results_csv", (best_seg_run / "results.csv") if best_seg_run else None, "YOLO-seg results.csv")

    # --- dir1 trained weights (YOLO/RT-DETR/FRCNN) often stored on D ---
    yolo_best = first_existing(
        [
            diplom / "runs/detect/runs/dir1/dir1_yolov8s_img640/weights/best.pt",
            rglob_best(diplom, "best.pt", prefer_keywords=["dir1", "yolov8"]) if diplom.exists() else None,
        ]
    )
    add(items, "DIR1_yolo_best_pt", yolo_best, "YOLO детектор best.pt (dir1)")

    rtdetr_best = first_existing(
        [
            diplom / "runs/detect/runs/dir1/dir1_rtdetr_l_img640/weights/best.pt",
            rglob_best(diplom, "best.pt", prefer_keywords=["dir1", "rtdetr"]) if diplom.exists() else None,
        ]
    )
    add(items, "DIR1_rtdetr_best_pt", rtdetr_best, "RT-DETR best.pt (dir1)")

    frcnn_weights = first_existing(
        [
            repo / "runs/dir1_detectron_frcnn/model_final.pth",
            rglob_best(repo, "model_final.pth", prefer_keywords=["dir1", "frcnn", "detectron"]),
        ]
    )
    add(items, "DIR1_detectron_frcnn_model_final", frcnn_weights, "Detectron2 Faster R-CNN weights (dir1)")

    # --- write outputs ---
    out_json = out_dir / "paths.json"
    out_md = out_dir / "paths.md"

    data = {it.key: {"status": it.status, "path": it.path, "note": it.note} for it in items}
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown
    lines = ["# Найденные пути (scan)\n"]
    for it in items:
        lines.append(f"## {it.key}")
        lines.append(f"- status: **{it.status}**")
        lines.append(f"- path: `{it.path}`" if it.path else "- path: `—`")
        if it.note:
            lines.append(f"- note: {it.note}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # console
    print("=== PATH SCAN DONE ===")
    print("repo:", repo)
    print("diplom:", diplom)
    print("saved:", out_json)
    missing = [it for it in items if it.status == "MISSING"]
    if missing:
        print("\nMISSING ITEMS:")
        for it in missing:
            print("-", it.key, "|", it.note)
    else:
        print("\nAll required items FOUND.")


if __name__ == "__main__":
    main()