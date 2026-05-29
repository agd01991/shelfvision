from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_BASE_NIGHT_ROOT = Path("/mnt/d/1Diplom/shelfvision_results/night_sku110k_v2_2026-05-28_00-16-10")
DEFAULT_WEIGHTS = Path("/mnt/d/1Diplom/runs/detect/runs/dir1/dir1_yolov8s_img640/weights/best.pt")
DEFAULT_RESULTS_PARENT = Path("/mnt/d/1Diplom/shelfvision_results")
DEFAULT_GALLERY_PARENT = Path("/mnt/d/1Diplom")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    gallery_build_mode: str
    dedup_threshold: float
    cluster_merge_threshold: float
    cluster_strong_merge_threshold: float
    cluster_min_similarity: float
    cluster_max_candidates: int
    max_refs_per_sku: int


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        name="01_greedy_dedup082_ref10",
        gallery_build_mode="greedy",
        dedup_threshold=0.82,
        cluster_merge_threshold=0.82,
        cluster_strong_merge_threshold=0.88,
        cluster_min_similarity=0.72,
        cluster_max_candidates=0,
        max_refs_per_sku=10,
    ),
    ExperimentConfig(
        name="02_cluster_merge082_min072_ref10",
        gallery_build_mode="cluster",
        dedup_threshold=0.82,
        cluster_merge_threshold=0.82,
        cluster_strong_merge_threshold=0.88,
        cluster_min_similarity=0.72,
        cluster_max_candidates=0,
        max_refs_per_sku=10,
    ),
    ExperimentConfig(
        name="03_cluster_merge084_min074_ref10",
        gallery_build_mode="cluster",
        dedup_threshold=0.82,
        cluster_merge_threshold=0.84,
        cluster_strong_merge_threshold=0.90,
        cluster_min_similarity=0.74,
        cluster_max_candidates=0,
        max_refs_per_sku=10,
    ),
    ExperimentConfig(
        name="04_cluster_merge086_min076_ref10",
        gallery_build_mode="cluster",
        dedup_threshold=0.82,
        cluster_merge_threshold=0.86,
        cluster_strong_merge_threshold=0.92,
        cluster_min_similarity=0.76,
        cluster_max_candidates=0,
        max_refs_per_sku=10,
    ),
    ExperimentConfig(
        name="05_cluster_merge084_min074_ref20",
        gallery_build_mode="cluster",
        dedup_threshold=0.82,
        cluster_merge_threshold=0.84,
        cluster_strong_merge_threshold=0.90,
        cluster_min_similarity=0.74,
        cluster_max_candidates=0,
        max_refs_per_sku=20,
    ),
]


SUMMARY_COLUMNS = [
    "experiment",
    "status",
    "model",
    "weights_key",
    "weights",
    "conf",
    "imgsz",
    "gallery_count",
    "query_count",
    "max_sku",
    "dedup_threshold",
    "max_refs_per_sku",
    "min_crop",
    "padding",
    "gallery_build_mode",
    "cluster_merge_threshold",
    "cluster_strong_merge_threshold",
    "cluster_min_similarity",
    "cluster_max_candidates",
    "query_objects",
    "matched",
    "unknown",
    "matched_rate",
    "unknown_rate",
    "avg_similarity",
    "created_demo_sku",
    "extracted_gallery_crops",
    "gallery_refs",
    "duplicate_refs",
    "skipped_duplicate_crops",
    "elapsed_seconds",
    "out_dir",
    "log_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SKU110K greedy vs clustered demo gallery comparison in Python."
    )
    parser.add_argument("--base-night-root", type=Path, default=DEFAULT_BASE_NIGHT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=None)
    parser.add_argument("--gallery-split", default="gallery_120")
    parser.add_argument("--query-split", default="query_140")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "0"))
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("IMG_SIZE", "640")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--run-id", default=os.environ.get("CLUSTER_RUN_ID", time.strftime("%Y-%m-%d_%H-%M-%S")))
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--gallery-root", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable, help="Python executable for child commands. Default: current interpreter")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="Timeout per experiment. 0 means no timeout")
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing output dirs before each experiment")
    return parser.parse_args()


def project_root() -> Path:
    return Path.cwd().resolve()


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def check_python_packages() -> None:
    required = ["pandas", "numpy", "cv2", "ultralytics"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing Python packages: {', '.join(missing)}")


def preflight(args: argparse.Namespace, gallery_images_dir: Path, query_images_dir: Path) -> None:
    root = project_root()
    print("=== SKU110K cluster comparison preflight ===", flush=True)
    print(f"Project root: {root}", flush=True)
    print(f"Base night root: {args.base_night_root}", flush=True)
    print(f"Gallery split: {gallery_images_dir}", flush=True)
    print(f"Query split: {query_images_dir}", flush=True)
    print(f"Weights: {args.weights}", flush=True)
    print(f"Python: {args.python}", flush=True)
    print("", flush=True)

    if not (root / "run_full_photo_identification_pipeline.py").exists():
        raise FileNotFoundError("run_full_photo_identification_pipeline.py not found. Run from shelfvision root.")
    if not gallery_images_dir.is_dir():
        raise FileNotFoundError(f"Gallery split not found: {gallery_images_dir}")
    if not query_images_dir.is_dir():
        raise FileNotFoundError(f"Query split not found: {query_images_dir}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"YOLOv8s weights not found: {args.weights}")

    check_python_packages()
    print("Python packages OK", flush=True)
    print(f"Gallery images: {count_images(gallery_images_dir)}", flush=True)
    print(f"Query images: {count_images(query_images_dir)}", flush=True)

    try:
        result = subprocess.run(["df", "-h", "/mnt/d"], check=False, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, flush=True)
    except Exception:
        pass

    print("Preflight OK", flush=True)
    print("", flush=True)


def init_summary(summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()


def build_command(
    args: argparse.Namespace,
    exp: ExperimentConfig,
    gallery_images_dir: Path,
    query_images_dir: Path,
    out_dir: Path,
    gallery_dir: Path,
) -> list[str]:
    return [
        args.python,
        "run_full_photo_identification_pipeline.py",
        "--model",
        "yolo",
        "--weights",
        str(args.weights),
        "--gallery-images-dir",
        str(gallery_images_dir),
        "--query-images-dir",
        str(query_images_dir),
        "--out-dir",
        str(out_dir),
        "--gallery-dir",
        str(gallery_dir),
        "--gallery-csv",
        str(gallery_dir / "gallery.csv"),
        "--gallery-limit",
        "0",
        "--query-limit",
        "0",
        "--conf",
        "0.25",
        "--imgsz",
        str(args.imgsz),
        "--device",
        str(args.device),
        "--max-sku",
        "150",
        "--min-score",
        "0.35",
        "--min-width",
        "20",
        "--min-height",
        "20",
        "--padding",
        "0.05",
        "--prefix",
        "sku_demo_",
        "--gallery-build-mode",
        exp.gallery_build_mode,
        "--dedup-threshold",
        str(exp.dedup_threshold),
        "--cluster-merge-threshold",
        str(exp.cluster_merge_threshold),
        "--cluster-strong-merge-threshold",
        str(exp.cluster_strong_merge_threshold),
        "--cluster-min-similarity",
        str(exp.cluster_min_similarity),
        "--cluster-pair-report-threshold",
        "0.75",
        "--cluster-max-candidates",
        str(exp.cluster_max_candidates),
        "--max-refs-per-sku",
        str(exp.max_refs_per_sku),
        "--threshold",
        "0.65",
        "--thresholds",
        "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        "--top-k",
        "3",
        "--visualize-limit",
        "60",
        "--progress-every",
        "25",
        "--shuffle",
        "--seed",
        str(args.seed),
        "--resume",
        "--skip-existing",
        "--no-visualize-inference",
    ]


def append_summary(
    summary_csv: Path,
    exp: ExperimentConfig,
    status: str,
    args: argparse.Namespace,
    gallery_count: int,
    query_count: int,
    out_dir: Path,
    log_file: Path,
    elapsed_seconds: int,
) -> None:
    full = read_json(out_dir / "05_reports" / "full_experiment_summary.json")
    demo = read_json(out_dir / "02_demo_gallery" / "demo_sku_gallery_summary.json")

    final_status = status
    if status == "ok" and not full:
        final_status = "ok_but_missing_summary"

    row = {
        "experiment": exp.name,
        "status": final_status,
        "model": "yolo",
        "weights_key": "yolov8s",
        "weights": str(args.weights),
        "conf": "0.25",
        "imgsz": args.imgsz,
        "gallery_count": gallery_count,
        "query_count": query_count,
        "max_sku": 150,
        "dedup_threshold": exp.dedup_threshold,
        "max_refs_per_sku": exp.max_refs_per_sku,
        "min_crop": 20,
        "padding": 0.05,
        "gallery_build_mode": exp.gallery_build_mode,
        "cluster_merge_threshold": exp.cluster_merge_threshold,
        "cluster_strong_merge_threshold": exp.cluster_strong_merge_threshold,
        "cluster_min_similarity": exp.cluster_min_similarity,
        "cluster_max_candidates": exp.cluster_max_candidates,
        "query_objects": full.get("query_objects_count", ""),
        "matched": full.get("matched", ""),
        "unknown": full.get("unknown", ""),
        "matched_rate": full.get("matched_rate", ""),
        "unknown_rate": full.get("unknown_rate", ""),
        "avg_similarity": full.get("avg_similarity", ""),
        "created_demo_sku": full.get("created_demo_sku_count", demo.get("created_sku_count", "")),
        "extracted_gallery_crops": full.get("extracted_gallery_crops_count", demo.get("extracted_crops_count", "")),
        "gallery_refs": demo.get("gallery_refs_count", ""),
        "duplicate_refs": demo.get("duplicate_refs_count", ""),
        "skipped_duplicate_crops": demo.get("skipped_duplicate_crops_count", ""),
        "elapsed_seconds": elapsed_seconds,
        "out_dir": str(out_dir),
        "log_file": str(log_file),
    }

    with summary_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writerow(row)


def run_experiment(
    args: argparse.Namespace,
    exp: ExperimentConfig,
    results_root: Path,
    gallery_root: Path,
    log_dir: Path,
    commands_dir: Path,
    summary_csv: Path,
    gallery_images_dir: Path,
    query_images_dir: Path,
) -> None:
    out_dir = results_root / exp.name
    gallery_dir = gallery_root / exp.name
    log_file = log_dir / f"{exp.name}.log"
    command_file = commands_dir / f"{exp.name}.txt"

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"RUN: {exp.name}", flush=True)
    print(
        "mode={mode} dedup={dedup} merge={merge} strong={strong} min_cluster={min_cluster} "
        "candidates={candidates} refs={refs}".format(
            mode=exp.gallery_build_mode,
            dedup=exp.dedup_threshold,
            merge=exp.cluster_merge_threshold,
            strong=exp.cluster_strong_merge_threshold,
            min_cluster=exp.cluster_min_similarity,
            candidates=exp.cluster_max_candidates,
            refs=exp.max_refs_per_sku,
        ),
        flush=True,
    )
    print(f"out={out_dir}", flush=True)
    print(f"log={log_file}", flush=True)
    print("=" * 60, flush=True)

    if not args.no_clean:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(gallery_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args, exp, gallery_images_dir, query_images_dir, out_dir, gallery_dir)
    command_file.write_text(" ".join(command) + "\n", encoding="utf-8")

    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_file.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(command) + "\n\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=project_root(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=args.timeout_seconds if args.timeout_seconds > 0 else None,
            )
            return_code = result.returncode
            status = "ok" if return_code == 0 else f"failed_{return_code}"
        except subprocess.TimeoutExpired:
            return_code = 124
            status = "timeout"
            log.write("\nTIMEOUT\n")

    elapsed = int(time.perf_counter() - started)
    if status == "ok":
        print(f"DONE: {exp.name} elapsed={elapsed}s", flush=True)
    else:
        print(f"FAILED: {exp.name} status={status} code={return_code} elapsed={elapsed}s", flush=True)

    append_summary(
        summary_csv=summary_csv,
        exp=exp,
        status=status,
        args=args,
        gallery_count=count_images(gallery_images_dir),
        query_count=count_images(query_images_dir),
        out_dir=out_dir,
        log_file=log_file,
        elapsed_seconds=elapsed,
    )


def run_reports(args: argparse.Namespace, results_root: Path, summary_csv: Path) -> None:
    if args.skip_reports:
        return
    report_script = project_root() / "run_night_experiments_report.py"
    if not report_script.exists():
        print(f"WARNING: report script not found: {report_script}", flush=True)
        return
    command = [
        args.python,
        str(report_script),
        "--results-root",
        str(results_root),
        "--summary-csv",
        str(summary_csv),
        "--out-dir",
        str(results_root),
        "--top-n",
        "20",
    ]
    print("Generating analytical reports...", flush=True)
    subprocess.run(command, cwd=project_root(), check=False)


def to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except Exception:
        return 0.0


def write_comparison_markdown(summary_csv: Path, output_md: Path, results_root: Path) -> None:
    if not summary_csv.exists():
        return
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: (to_float(row, "matched_rate"), to_float(row, "avg_similarity")), reverse=True)

    lines = [
        "# SKU110K greedy vs clustered gallery comparison",
        "",
        f"Results root: `{results_root}`",
        "",
        "| # | experiment | mode | matched_rate | unknown_rate | avg_similarity | demo_sku | gallery_refs | merge | strong | min_cluster | refs |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | `{row.get('experiment', '')}` | {row.get('gallery_build_mode', '')} | "
            f"{to_float(row, 'matched_rate'):.4f} | {to_float(row, 'unknown_rate'):.4f} | "
            f"{to_float(row, 'avg_similarity'):.4f} | {row.get('created_demo_sku', '')} | "
            f"{row.get('gallery_refs', '')} | {row.get('cluster_merge_threshold', '')} | "
            f"{row.get('cluster_strong_merge_threshold', '')} | {row.get('cluster_min_similarity', '')} | "
            f"{row.get('max_refs_per_sku', '')} |"
        )
    lines.extend(
        [
            "",
            "## Как выбирать результат",
            "",
            "Сначала сравни `matched_rate` и `avg_similarity`, затем обязательно проверь `02_demo_gallery/cluster_contact_sheets` у cluster-запусков. Если внутри одного contact sheet склеены разные товары, подними `cluster_merge_threshold` или `cluster_min_similarity`.",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Cluster comparison markdown saved: {output_md}", flush=True)


def main() -> None:
    args = parse_args()
    splits_root = args.splits_root or (args.base_night_root / "_splits")
    gallery_images_dir = splits_root / args.gallery_split
    query_images_dir = splits_root / args.query_split

    results_root = args.results_root or (DEFAULT_RESULTS_PARENT / f"cluster_compare_sku110k_{args.run_id}")
    gallery_root = args.gallery_root or (DEFAULT_GALLERY_PARENT / f"sku_gallery_cluster_compare_sku110k_{args.run_id}")
    log_dir = results_root / "_logs"
    commands_dir = results_root / "_commands"
    summary_csv = results_root / "night_experiments_summary.csv"
    summary_md = results_root / "cluster_comparison_summary.md"

    results_root.mkdir(parents=True, exist_ok=True)
    gallery_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)

    preflight(args, gallery_images_dir, query_images_dir)
    init_summary(summary_csv)

    for exp in EXPERIMENTS:
        run_experiment(
            args=args,
            exp=exp,
            results_root=results_root,
            gallery_root=gallery_root,
            log_dir=log_dir,
            commands_dir=commands_dir,
            summary_csv=summary_csv,
            gallery_images_dir=gallery_images_dir,
            query_images_dir=query_images_dir,
        )

    run_reports(args, results_root, summary_csv)
    write_comparison_markdown(summary_csv, summary_md, results_root)

    print("", flush=True)
    print("=== CLUSTER COMPARISON FINISHED ===", flush=True)
    print(f"Results root: {results_root}", flush=True)
    print(f"Summary CSV: {summary_csv}", flush=True)
    print(f"Summary MD: {summary_md}", flush=True)
    print(f"Logs: {log_dir}", flush=True)


if __name__ == "__main__":
    main()
