#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[ERROR] $(date -Is) line=${LINENO} cmd=${BASH_COMMAND}" >&2' ERR INT TERM

# ============================================================
# ShelfVision SKU110K: greedy vs clustered gallery comparison
# ============================================================
# Запускать из корня проекта shelfvision в активированном .venv_wsl.
# Скрипт переиспользует фиксированные split-папки из ночной серии, чтобы
# сравнение greedy/cluster было честным на одинаковых gallery/query.
# ============================================================

PROJECT_ROOT="$(pwd)"

BASE_NIGHT_ROOT="${BASE_NIGHT_ROOT:-/mnt/d/1Diplom/shelfvision_results/night_sku110k_v2_2026-05-28_00-16-10}"
SPLITS_ROOT="${SPLITS_ROOT:-${BASE_NIGHT_ROOT}/_splits}"
GALLERY_SPLIT="${GALLERY_SPLIT:-gallery_120}"
QUERY_SPLIT="${QUERY_SPLIT:-query_140}"

DEVICE="${DEVICE:-0}"
IMG_SIZE="${IMG_SIZE:-640}"
SEED="${SEED:-42}"
YOLOV8S_WEIGHTS="${YOLOV8S_WEIGHTS:-/mnt/d/1Diplom/runs/detect/runs/dir1/dir1_yolov8s_img640/weights/best.pt}"

RUN_ID="${CLUSTER_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/d/1Diplom/shelfvision_results/cluster_compare_sku110k_${RUN_ID}}"
GALLERY_ROOT="${GALLERY_ROOT:-/mnt/d/1Diplom/sku_gallery_cluster_compare_sku110k_${RUN_ID}}"
LOG_DIR="${RESULTS_ROOT}/_logs"
COMMANDS_DIR="${RESULTS_ROOT}/_commands"
SUMMARY_CSV="${RESULTS_ROOT}/night_experiments_summary.csv"
SUMMARY_MD="${RESULTS_ROOT}/cluster_comparison_summary.md"

mkdir -p "$RESULTS_ROOT" "$GALLERY_ROOT" "$LOG_DIR" "$COMMANDS_DIR"

GALLERY_IMAGES_DIR="${SPLITS_ROOT}/${GALLERY_SPLIT}"
QUERY_IMAGES_DIR="${SPLITS_ROOT}/${QUERY_SPLIT}"

check_preflight() {
  echo "=== SKU110K cluster comparison preflight ==="
  echo "Project root: $PROJECT_ROOT"
  echo "Base night root: $BASE_NIGHT_ROOT"
  echo "Gallery split: $GALLERY_IMAGES_DIR"
  echo "Query split: $QUERY_IMAGES_DIR"
  echo "Results root: $RESULTS_ROOT"
  echo "Gallery root: $GALLERY_ROOT"
  echo

  if [ ! -f "run_full_photo_identification_pipeline.py" ]; then
    echo "ERROR: run_full_photo_identification_pipeline.py not found. Run from shelfvision root."
    exit 1
  fi
  if [ ! -d "$GALLERY_IMAGES_DIR" ]; then
    echo "ERROR: gallery split not found: $GALLERY_IMAGES_DIR"
    exit 1
  fi
  if [ ! -d "$QUERY_IMAGES_DIR" ]; then
    echo "ERROR: query split not found: $QUERY_IMAGES_DIR"
    exit 1
  fi
  if [ ! -f "$YOLOV8S_WEIGHTS" ]; then
    echo "ERROR: YOLOv8s weights not found: $YOLOV8S_WEIGHTS"
    exit 1
  fi

  python - <<'PY'
import importlib.util
import sys
missing = [name for name in ["pandas", "numpy", "cv2", "ultralytics"] if importlib.util.find_spec(name) is None]
if missing:
    print("ERROR: missing packages:", ", ".join(missing))
    sys.exit(1)
print("Python packages OK")
PY

  df -h /mnt/d || true
  echo "Preflight OK"
}

write_summary_header() {
  cat > "$SUMMARY_CSV" <<CSV
experiment,status,model,weights_key,weights,conf,imgsz,gallery_count,query_count,max_sku,dedup_threshold,max_refs_per_sku,min_crop,padding,gallery_build_mode,cluster_merge_threshold,cluster_strong_merge_threshold,cluster_min_similarity,cluster_max_candidates,query_objects,matched,unknown,matched_rate,unknown_rate,avg_similarity,created_demo_sku,extracted_gallery_crops,gallery_refs,duplicate_refs,skipped_duplicate_crops,elapsed_seconds,out_dir,log_file
CSV
}

append_summary() {
  local name="$1"
  local status="$2"
  local conf="$3"
  local max_sku="$4"
  local dedup_threshold="$5"
  local max_refs="$6"
  local min_crop="$7"
  local padding="$8"
  local gallery_build_mode="$9"
  local cluster_merge_threshold="${10}"
  local cluster_strong_merge_threshold="${11}"
  local cluster_min_similarity="${12}"
  local cluster_max_candidates="${13}"
  local out_dir="${14}"
  local log_file="${15}"
  local elapsed="${16}"

  python - "$SUMMARY_CSV" "$name" "$status" "$YOLOV8S_WEIGHTS" "$conf" "$IMG_SIZE" "$max_sku" "$dedup_threshold" "$max_refs" "$min_crop" "$padding" "$gallery_build_mode" "$cluster_merge_threshold" "$cluster_strong_merge_threshold" "$cluster_min_similarity" "$cluster_max_candidates" "$out_dir" "$log_file" "$elapsed" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    summary_csv,
    name,
    status,
    weights,
    conf,
    imgsz,
    max_sku,
    dedup_threshold,
    max_refs,
    min_crop,
    padding,
    gallery_build_mode,
    cluster_merge_threshold,
    cluster_strong_merge_threshold,
    cluster_min_similarity,
    cluster_max_candidates,
    out_dir,
    log_file,
    elapsed,
) = sys.argv[1:]

out_dir = Path(out_dir)

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

full = read_json(out_dir / "05_reports" / "full_experiment_summary.json")
demo = read_json(out_dir / "02_demo_gallery" / "demo_sku_gallery_summary.json")

row = {
    "experiment": name,
    "status": status if full or status != "ok" else "ok_but_missing_summary",
    "model": "yolo",
    "weights_key": "yolov8s",
    "weights": weights,
    "conf": conf,
    "imgsz": imgsz,
    "gallery_count": 120,
    "query_count": 140,
    "max_sku": max_sku,
    "dedup_threshold": dedup_threshold,
    "max_refs_per_sku": max_refs,
    "min_crop": min_crop,
    "padding": padding,
    "gallery_build_mode": gallery_build_mode,
    "cluster_merge_threshold": cluster_merge_threshold,
    "cluster_strong_merge_threshold": cluster_strong_merge_threshold,
    "cluster_min_similarity": cluster_min_similarity,
    "cluster_max_candidates": cluster_max_candidates,
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
    "elapsed_seconds": elapsed,
    "out_dir": str(out_dir),
    "log_file": log_file,
}

fieldnames = list(row.keys())
with open(summary_csv, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(row)
PY
}

run_exp() {
  local name="$1"
  local gallery_build_mode="$2"
  local dedup_threshold="$3"
  local cluster_merge_threshold="$4"
  local cluster_strong_merge_threshold="$5"
  local cluster_min_similarity="$6"
  local cluster_max_candidates="$7"
  local max_refs="$8"

  local conf="0.25"
  local max_sku="150"
  local min_crop="20"
  local padding="0.05"
  local out_dir="${RESULTS_ROOT}/${name}"
  local gallery_dir="${GALLERY_ROOT}/${name}"
  local log_file="${LOG_DIR}/${name}.log"
  local command_file="${COMMANDS_DIR}/${name}.txt"

  echo
  echo "============================================================"
  echo "RUN: $name"
  echo "mode=$gallery_build_mode dedup=$dedup_threshold merge=$cluster_merge_threshold strong=$cluster_strong_merge_threshold min_cluster=$cluster_min_similarity candidates=$cluster_max_candidates refs=$max_refs"
  echo "out=$out_dir"
  echo "log=$log_file"
  echo "============================================================"

  rm -rf "$out_dir" "$gallery_dir"
  mkdir -p "$out_dir" "$gallery_dir"

  cat > "$command_file" <<CMD
PYTHONUNBUFFERED=1 python run_full_photo_identification_pipeline.py \\
  --model yolo \\
  --weights "$YOLOV8S_WEIGHTS" \\
  --gallery-images-dir "$GALLERY_IMAGES_DIR" \\
  --query-images-dir "$QUERY_IMAGES_DIR" \\
  --out-dir "$out_dir" \\
  --gallery-dir "$gallery_dir" \\
  --gallery-csv "$gallery_dir/gallery.csv" \\
  --gallery-limit 0 \\
  --query-limit 0 \\
  --conf "$conf" \\
  --imgsz "$IMG_SIZE" \\
  --device "$DEVICE" \\
  --max-sku "$max_sku" \\
  --min-score 0.35 \\
  --min-width "$min_crop" \\
  --min-height "$min_crop" \\
  --padding "$padding" \\
  --prefix sku_demo_ \\
  --gallery-build-mode "$gallery_build_mode" \\
  --dedup-threshold "$dedup_threshold" \\
  --cluster-merge-threshold "$cluster_merge_threshold" \\
  --cluster-strong-merge-threshold "$cluster_strong_merge_threshold" \\
  --cluster-min-similarity "$cluster_min_similarity" \\
  --cluster-pair-report-threshold 0.75 \\
  --cluster-max-candidates "$cluster_max_candidates" \\
  --max-refs-per-sku "$max_refs" \\
  --threshold 0.65 \\
  --thresholds 0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90 \\
  --top-k 3 \\
  --visualize-limit 60 \\
  --progress-every 25 \\
  --shuffle \\
  --seed "$SEED" \\
  --resume \\
  --skip-existing \\
  --no-visualize-inference
CMD

  local started
  started="$(date +%s)"
  set +e
  bash "$command_file" > "$log_file" 2>&1
  local code=$?
  set -e
  local finished
  finished="$(date +%s)"
  local elapsed=$((finished - started))

  if [ "$code" -eq 0 ]; then
    echo "DONE: $name elapsed=${elapsed}s"
    append_summary "$name" "ok" "$conf" "$max_sku" "$dedup_threshold" "$max_refs" "$min_crop" "$padding" "$gallery_build_mode" "$cluster_merge_threshold" "$cluster_strong_merge_threshold" "$cluster_min_similarity" "$cluster_max_candidates" "$out_dir" "$log_file" "$elapsed"
  else
    echo "FAILED: $name code=$code elapsed=${elapsed}s"
    append_summary "$name" "failed_${code}" "$conf" "$max_sku" "$dedup_threshold" "$max_refs" "$min_crop" "$padding" "$gallery_build_mode" "$cluster_merge_threshold" "$cluster_strong_merge_threshold" "$cluster_min_similarity" "$cluster_max_candidates" "$out_dir" "$log_file" "$elapsed"
  fi
}

generate_reports() {
  python run_night_experiments_report.py \
    --results-root "$RESULTS_ROOT" \
    --summary-csv "$SUMMARY_CSV" \
    --out-dir "$RESULTS_ROOT" \
    --top-n 20 || true

  python - "$SUMMARY_CSV" "$SUMMARY_MD" "$RESULTS_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
root = Path(sys.argv[3])
rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))

def f(row, key):
    try:
        return float(row.get(key, "") or 0)
    except Exception:
        return 0.0

rows = sorted(rows, key=lambda r: (f(r, "matched_rate"), f(r, "avg_similarity")), reverse=True)
lines = [
    "# SKU110K greedy vs clustered gallery comparison",
    "",
    f"Results root: `{root}`",
    "",
    "| # | experiment | mode | matched_rate | unknown_rate | avg_similarity | demo_sku | gallery_refs | merge | strong | min_cluster | refs |",
    "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for idx, row in enumerate(rows, start=1):
    lines.append(
        f"| {idx} | `{row.get('experiment','')}` | {row.get('gallery_build_mode','')} | "
        f"{f(row,'matched_rate'):.4f} | {f(row,'unknown_rate'):.4f} | {f(row,'avg_similarity'):.4f} | "
        f"{row.get('created_demo_sku','')} | {row.get('gallery_refs','')} | {row.get('cluster_merge_threshold','')} | "
        f"{row.get('cluster_strong_merge_threshold','')} | {row.get('cluster_min_similarity','')} | {row.get('max_refs_per_sku','')} |"
    )
lines.extend([
    "",
    "## Как выбирать результат",
    "",
    "Сначала сравни `matched_rate` и `avg_similarity`, затем обязательно проверь `02_demo_gallery/cluster_contact_sheets` у cluster-запусков. Если внутри одного contact sheet склеены разные товары, подними `cluster_merge_threshold` или `cluster_min_similarity`.",
])
md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Cluster comparison markdown saved: {md_path}")
PY
}

main() {
  check_preflight
  write_summary_header

  # 01: greedy baseline, близкий к текущему лучшему 09 из ночной серии.
  run_exp "01_greedy_dedup082_ref10" "greedy" "0.82" "0.82" "0.88" "0.72" "0" "10"

  # 02-05: clustered mode, подбор строгости объединения.
  run_exp "02_cluster_merge082_min072_ref10" "cluster" "0.82" "0.82" "0.88" "0.72" "0" "10"
  run_exp "03_cluster_merge084_min074_ref10" "cluster" "0.82" "0.84" "0.90" "0.74" "0" "10"
  run_exp "04_cluster_merge086_min076_ref10" "cluster" "0.82" "0.86" "0.92" "0.76" "0" "10"

  # 05: больше refs на SKU, если contact sheets выглядят чисто.
  run_exp "05_cluster_merge084_min074_ref20" "cluster" "0.82" "0.84" "0.90" "0.74" "0" "20"

  generate_reports

  echo
  echo "=== CLUSTER COMPARISON FINISHED ==="
  echo "Results root: $RESULTS_ROOT"
  echo "Summary CSV: $SUMMARY_CSV"
  echo "Summary MD: $SUMMARY_MD"
  echo "Logs: $LOG_DIR"
}

main "$@"
