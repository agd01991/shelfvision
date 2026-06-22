from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

import action_history
from path_utils import to_current_os_path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class UserImageManifestRow:
    split: str
    index: int
    image_path: str
    image_name: str


def _p(value: str | Path | None) -> Path:
    return to_current_os_path(value)


def _read_json(path: str | Path) -> Dict[str, Any]:
    path = _p(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _image_files(path: str | Path, limit: int = 0) -> List[Path]:
    root = _p(path)
    if not root.exists():
        return []
    files = sorted(
        item for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )
    if limit and limit > 0:
        return files[:limit]
    return files


def _default_weight(config: Dict[str, Any], model: str = "yolo") -> str:
    weights = config.get("weights", {}) if isinstance(config, dict) else {}
    value = str(weights.get(model, "") or "").strip()
    return value


def _full_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("full_photo_identification", {}) or {})


def _runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("runtime", {}) or {})


def _gallery_from_summary(exp: Path) -> tuple[Path | None, Path | None]:
    exp = _p(exp)
    candidates = [
        exp / "02_demo_gallery" / "demo_sku_gallery_summary.json",
        exp / "05_reports" / "full_experiment_summary.json",
        exp / "05_reports" / "existing_identification_summary.json",
    ]
    for summary_path in candidates:
        raw = _read_json(summary_path)
        if not raw:
            continue
        csv_value = raw.get("gallery_csv") or raw.get("output_gallery_csv")
        dir_value = raw.get("gallery_dir") or raw.get("output_gallery_dir")
        csv_path = _p(str(csv_value)) if csv_value else None
        dir_path = _p(str(dir_value)) if dir_value else None
        if csv_path and csv_path.exists():
            return csv_path, dir_path if dir_path and dir_path.exists() else csv_path.parent
    fallback = exp / "02_demo_gallery" / "sku_gallery_final" / "gallery.csv"
    if fallback.exists():
        return fallback, fallback.parent
    return None, None


def _config_gallery(config: Dict[str, Any]) -> tuple[Path | None, Path | None]:
    full = _full_profile(config)
    csv_value = str(full.get("gallery_csv", "") or "").strip()
    dir_value = str(full.get("gallery_dir", "") or "").strip()
    csv_path = _p(csv_value) if csv_value else None
    dir_path = _p(dir_value) if dir_value else None
    if csv_path and csv_path.exists():
        return csv_path, dir_path if dir_path and dir_path.exists() else csv_path.parent
    return None, None


def _current_gallery(exp: Path, config: Dict[str, Any]) -> tuple[Path | None, Path | None]:
    csv_path, dir_path = _gallery_from_summary(exp)
    if csv_path is not None:
        return csv_path, dir_path
    return _config_gallery(config)


def _write_query_manifest(out_dir: Path, images: List[Path]) -> None:
    manifest_dir = _p(out_dir) / "00_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        UserImageManifestRow("query", index, str(path), path.name)
        for index, path in enumerate(images, start=1)
    ]
    df = pd.DataFrame([asdict(row) for row in rows])
    df.to_csv(manifest_dir / "all_images.csv", index=False)
    df.to_csv(manifest_dir / "query_images.csv", index=False)
    pd.DataFrame(columns=df.columns).to_csv(manifest_dir / "gallery_images.csv", index=False)
    (manifest_dir / "split_params.json").write_text(
        json.dumps(
            {
                "split_level": "image",
                "split_method": "user_query_only",
                "gallery_images_count": 0,
                "query_images_count": len(images),
                "total_images_count": len(images),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_command(cmd: List[str], title: str) -> subprocess.CompletedProcess[str]:
    st.caption(" ".join(cmd))
    with st.spinner(title):
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    with st.expander(f"Вывод команды: {title}", expanded=result.returncode != 0):
        if result.stdout:
            st.code(result.stdout[-12000:], language="text")
        if result.stderr:
            st.code(result.stderr[-12000:], language="text")
    return result


def _inference_cmd(
    config: Dict[str, Any],
    model: str,
    weights: Path,
    user_images_dir: Path,
    out_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "run_inference.py"),
        "--model", model,
        "--weights", str(weights),
        "--images-dir", str(user_images_dir),
        "--out-dir", str(out_dir / "03_query_inference"),
        "--conf", str(conf),
        "--imgsz", str(imgsz),
    ]
    if device.strip():
        cmd.extend(["--device", device.strip()])
    return cmd


def _identify_cmd(
    config: Dict[str, Any],
    out_dir: Path,
    gallery_dir: Path,
    gallery_csv: Path,
    threshold: float,
    ambiguity_margin: float,
    top_k: int,
) -> List[str]:
    query_predictions = out_dir / "03_query_inference" / "predictions.json"
    cmd = [
        sys.executable,
        str(ROOT / "run_existing_photo_identification.py"),
        "--out-dir", str(out_dir),
        "--query-predictions-json", str(query_predictions),
        "--gallery-dir", str(gallery_dir),
        "--gallery-csv", str(gallery_csv),
        "--threshold", str(threshold),
        "--ambiguity-margin", str(ambiguity_margin),
        "--top-k", str(top_k),
        "--visualize-limit", "80",
        "--progress-every", "25",
    ]
    if bool(_full_profile(config).get("enable_uncertain_status", True)):
        cmd.append("--enable-uncertain-status")
    return cmd


def _full_pipeline_cmd(
    config: Dict[str, Any],
    user_images_dir: Path,
    out_dir: Path,
    gallery_dir: Path,
    gallery_csv: Path,
    model: str,
    weights: Path,
    gallery_count: int,
    query_count: int,
    max_sku: int,
    max_refs: int,
    conf: float,
    imgsz: int,
    device: str,
    threshold: float,
    ambiguity_margin: float,
    top_k: int,
    gallery_build_mode: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "run_full_photo_identification_pipeline.py"),
        "--model", model,
        "--weights", str(weights),
        "--images-dir", str(user_images_dir),
        "--out-dir", str(out_dir),
        "--gallery-dir", str(gallery_dir),
        "--gallery-csv", str(gallery_csv),
        "--gallery-count", str(gallery_count),
        "--query-count", str(query_count),
        "--max-sku", str(max_sku),
        "--max-refs-per-sku", str(max_refs),
        "--conf", str(conf),
        "--imgsz", str(imgsz),
        "--threshold", str(threshold),
        "--ambiguity-margin", str(ambiguity_margin),
        "--top-k", str(top_k),
        "--prefix", "user_sku_",
        "--gallery-build-mode", gallery_build_mode,
        "--visualize-limit", "80",
        "--progress-every", "25",
        "--shuffle",
        "--resume",
        "--skip-existing",
    ]
    if device.strip():
        cmd.extend(["--device", device.strip()])
    if bool(_full_profile(config).get("enable_uncertain_status", True)):
        cmd.append("--enable-uncertain-status")
    return cmd


def _combine_gallery_csv(existing_csv: Path, user_csv: Path, output_csv: Path) -> Path:
    existing = pd.read_csv(_p(existing_csv)).fillna("")
    user = pd.read_csv(_p(user_csv)).fillna("")
    user = user.copy()
    if "sku_id" in user.columns:
        user["sku_id"] = user["sku_id"].astype(str).map(
            lambda value: value if value.startswith("user_") else f"user_{value}"
        )
    if "sku_name" in user.columns:
        user["sku_name"] = user["sku_name"].astype(str).map(
            lambda value: value if value.startswith("user ") else f"user {value}"
        )
    combined = pd.concat([existing, user], ignore_index=True, sort=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return output_csv


def _render_result_actions(out_dir: Path) -> None:
    out_dir = _p(out_dir)
    st.success(f"Результат сохранён: `{out_dir}`")
    if st.button("Открыть этот результат в основном интерфейсе", use_container_width=True):
        st.session_state["demo_experiment_dir"] = str(out_dir)
        st.rerun()
    summary = out_dir / "05_reports" / "full_experiment_summary.json"
    existing_summary = out_dir / "05_reports" / "existing_identification_summary.json"
    for path in [summary, existing_summary]:
        if path.exists():
            with st.expander(f"Сводка: {path.name}", expanded=True):
                st.json(_read_json(path))
            break


def page_user_photos(config: Dict[str, Any], current_experiment_dir: str | Path) -> None:
    st.subheader("Свои фото стеллажей")
    st.caption(
        "Раздел запускает предобученную модель детекции на неразмеченных фотографиях пользователя "
        "и выполняет SKU-сопоставление по выбранной галерее."
    )

    current_exp = _p(current_experiment_dir)
    full = _full_profile(config)
    runtime = _runtime(config)

    user_images_dir = _p(
        st.text_input(
            "Папка с неразмеченными фотографиями пользователя",
            value=st.session_state.get("user_images_dir", ""),
            placeholder="например: D:/1Diplom/user_shelf_photos",
            key="user_images_dir_input",
        )
    )
    st.session_state["user_images_dir"] = str(user_images_dir)

    images = _image_files(user_images_dir)
    if user_images_dir.exists():
        st.info(f"Найдено изображений: {len(images)}")
    elif str(user_images_dir).strip() not in {"", "."}:
        st.warning(f"Папка не найдена: `{user_images_dir}`")

    mode = st.radio(
        "Режим работы",
        [
            "Фото пользователя + текущая SKU-галерея",
            "Только фото пользователя: авто-галерея + query",
            "Текущая галерея + авто-группы пользователя",
        ],
        horizontal=False,
        help=(
            "Первый режим использует готовую галерею текущего эксперимента. "
            "Второй строит галерею только из пользовательских фото. "
            "Третий сначала строит пользовательские группы, затем добавляет их к текущей галерее."
        ),
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        model = st.selectbox("Модель детекции", ["yolo", "yolo_seg", "rtdetr", "frcnn"], index=0)
        weights = _p(
            st.text_input(
                "Путь к весам модели",
                value=_default_weight(config, model) or _default_weight(config, "yolo"),
            )
        )
    with c2:
        conf = st.number_input("conf", 0.01, 0.99, float(runtime.get("conf", 0.25) or 0.25), step=0.01)
        imgsz = st.number_input("imgsz", 320, 1536, int(runtime.get("imgsz", 640) or 640), step=32)
        device = st.text_input("device", value=str(runtime.get("device", "") or ""))
    with c3:
        threshold = st.number_input("threshold τ", 0.10, 0.99, float(full.get("threshold", 0.65) or 0.65), step=0.01)
        ambiguity = st.number_input("margin δ", 0.0, 0.50, float(full.get("ambiguity_margin", 0.03) or 0.03), step=0.01)
        top_k = st.number_input("top-k", 1, 20, int(full.get("top_k", 5) or 5), step=1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_root = _p("D:/1Diplom/shelfvision_results/user_photo_demo")
    out_root = _p(st.text_input("Папка для результатов пользовательского запуска", value=str(default_root)))
    out_dir = out_root / f"run_{stamp}"

    gallery_csv, gallery_dir = _current_gallery(current_exp, config)
    if gallery_csv:
        st.caption(f"Текущая галерея: `{gallery_csv}`")
    else:
        st.warning("Текущая SKU-галерея не найдена. Используйте режим только по фотографиям пользователя.")

    gallery_count = 1
    query_count = 0
    max_sku = int(full.get("max_sku", 50) or 50)
    max_refs = int(full.get("max_refs_per_sku", 5) or 5)
    gallery_build_mode = str(full.get("gallery_build_mode", "greedy") or "greedy")
    if "Только фото пользователя" in mode or "авто-группы пользователя" in mode:
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            gallery_count = st.number_input("Сколько фото взять в галерею", 1, max(1, len(images)), min(10, max(1, len(images) // 2 or 1)))
        with c5:
            query_count = st.number_input("Сколько фото проверить", 0, max(0, len(images)), 0, help="0 = все оставшиеся")
        with c6:
            max_sku = st.number_input("Максимум user SKU", 1, 500, min(50, max(1, len(images) * 3)))
        with c7:
            max_refs = st.number_input("Эталонов на user SKU", 1, 50, min(5, max_refs))
        gallery_build_mode = st.selectbox("Сборка user-галереи", ["greedy", "cluster"], index=0)

    can_run = bool(images) and weights.exists()
    if not weights.exists():
        st.warning(f"Файл весов не найден: `{weights}`")

    if st.button("Запустить обработку пользовательских фото", type="primary", use_container_width=True, disabled=not can_run):
        out_dir.mkdir(parents=True, exist_ok=True)
        if mode == "Фото пользователя + текущая SKU-галерея":
            if gallery_csv is None or gallery_dir is None:
                st.error("Для этого режима нужна существующая SKU-галерея текущего эксперимента.")
                return
            _write_query_manifest(out_dir, images)
            result = _run_command(
                _inference_cmd(config, model, weights, user_images_dir, out_dir, float(conf), int(imgsz), device),
                "Детекция пользовательских фотографий",
            )
            if result.returncode != 0:
                st.error("Детекция завершилась ошибкой.")
                return
            result = _run_command(
                _identify_cmd(config, out_dir, gallery_dir, gallery_csv, float(threshold), float(ambiguity), int(top_k)),
                "Идентификация по текущей SKU-галерее",
            )
            if result.returncode != 0:
                st.error("Идентификация завершилась ошибкой.")
                return

        elif mode == "Только фото пользователя: авто-галерея + query":
            user_gallery_dir = out_dir / "02_demo_gallery" / "user_sku_gallery"
            user_gallery_csv = user_gallery_dir / "gallery.csv"
            result = _run_command(
                _full_pipeline_cmd(
                    config, user_images_dir, out_dir, user_gallery_dir, user_gallery_csv,
                    model, weights, int(gallery_count), int(query_count), int(max_sku), int(max_refs),
                    float(conf), int(imgsz), device, float(threshold), float(ambiguity), int(top_k), gallery_build_mode,
                ),
                "Полный пользовательский контур: авто-галерея + query",
            )
            if result.returncode != 0:
                st.error("Пользовательский полный контур завершился ошибкой.")
                return

        else:
            if gallery_csv is None or gallery_dir is None:
                st.error("Для комбинированного режима нужна существующая SKU-галерея текущего эксперимента.")
                return
            user_gallery_dir = out_dir / "02_demo_gallery" / "user_sku_gallery"
            user_gallery_csv = user_gallery_dir / "gallery.csv"
            result = _run_command(
                _full_pipeline_cmd(
                    config, user_images_dir, out_dir, user_gallery_dir, user_gallery_csv,
                    model, weights, int(gallery_count), int(query_count), int(max_sku), int(max_refs),
                    float(conf), int(imgsz), device, float(threshold), float(ambiguity), int(top_k), gallery_build_mode,
                ),
                "Построение пользовательских групп и query-предсказаний",
            )
            if result.returncode != 0:
                st.error("Построение пользовательских групп завершилось ошибкой.")
                return
            combined_csv = out_dir / "02_demo_gallery" / "combined_gallery.csv"
            _combine_gallery_csv(gallery_csv, user_gallery_csv, combined_csv)
            result = _run_command(
                _identify_cmd(config, out_dir, combined_csv.parent, combined_csv, float(threshold), float(ambiguity), int(top_k)),
                "Идентификация по текущей галерее и пользовательским группам",
            )
            if result.returncode != 0:
                st.error("Комбинированная идентификация завершилась ошибкой.")
                return

        action_history.append_event(
            current_exp,
            "user_photo_workflow",
            "Запущена обработка пользовательских фотографий",
            f"mode={mode}; out_dir={out_dir}",
            out_dir,
        )
        _render_result_actions(out_dir)

    recent_root = out_root if out_root.exists() else default_root
    if recent_root.exists():
        runs = sorted([path for path in recent_root.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        if runs:
            st.markdown("#### Последние пользовательские запуски")
            selected = st.selectbox("Открыть существующий запуск", runs, format_func=lambda p: p.name)
            if st.button("Открыть выбранный запуск в основном интерфейсе", use_container_width=True):
                st.session_state["demo_experiment_dir"] = str(selected)
                st.rerun()
