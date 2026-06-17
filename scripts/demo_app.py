from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import yaml

from panel_progress import CommandStep, run_steps_with_progress
from run_summary_panel import collect_summary


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "shelfvision.yaml"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STATUS_COLUMNS = ["sku_status", "status", "assignment_status"]
STATUS_LABELS = {
    "matched": "Уверенно идентифицировано",
    "matched_uncertain": "Требует проверки",
    "unknown": "Не определено",
}
CORRECTION_LABELS = {
    "confirm_match": "Подтверждён предложенный SKU",
    "change_sku": "Выбран другой SKU",
    "mark_unknown": "Оставлено не определённым",
    "needs_review": "Отложено для проверки",
}
CORRECTION_COLUMNS = [
    "created_at",
    "image_name",
    "object_id",
    "old_status",
    "old_sku_id",
    "old_sku_name",
    "new_sku_id",
    "correction_type",
    "comment",
    "sku_confidence",
    "distinct_margin",
    "second_distinct_sku",
    "crop_path",
]


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _demo_config() -> Dict[str, Any]:
    config = _load_config()
    demo = config.get("demo_app", {}) or {}
    return {
        "app_title": demo.get("app_title", "ShelfVision"),
        "output_root": demo.get("output_root", "results/demo_user_runs"),
        "weights": demo.get("weights", "models/shelfvision_detector.pt"),
        "system_gallery_dir": demo.get("system_gallery_dir", "system_gallery/sku_gallery_final"),
        "system_gallery_csv": demo.get("system_gallery_csv", "system_gallery/sku_gallery_final/gallery.csv"),
        "default_images_dir": demo.get("default_images_dir", ""),
        "model": demo.get("model", "yolo"),
        "conf": float(demo.get("conf", 0.25)),
        "imgsz": int(demo.get("imgsz", 640)),
        "device": str(demo.get("device", "")).strip(),
        "threshold": float(demo.get("threshold", 0.55)),
        "thresholds": str(demo.get("thresholds", "0.45,0.50,0.55,0.60,0.65")),
        "top_k": int(demo.get("top_k", 5)),
        "ambiguity_margin": float(demo.get("ambiguity_margin", 0.03)),
        "visualize_limit": int(demo.get("visualize_limit", 50)),
        "progress_every": int(demo.get("progress_every", 50)),
    }


def _to_runtime_path(value: str | Path) -> Path:
    text = str(value).strip().strip('"').strip("'")
    if re.match(r"^[A-Za-z]:[\\/]", text):
        drive = text[0].lower()
        rest = text[2:].lstrip("\\/").replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(text).expanduser()


def _display_path(path: str | Path) -> str:
    return str(path)


def _status_label(value: Any) -> str:
    return STATUS_LABELS.get(str(value), str(value))


def _correction_label(value: Any) -> str:
    return CORRECTION_LABELS.get(str(value), str(value))


def _status_column(df: pd.DataFrame) -> Optional[str]:
    for col in STATUS_COLUMNS:
        if col in df.columns:
            return col
    return None


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    status_col = _status_column(result)
    if status_col:
        result[status_col] = result[status_col].map(_status_label)
        result = result.rename(columns={status_col: "Статус"})
    if "correction_type" in result.columns:
        result["correction_type"] = result["correction_type"].map(_correction_label)
        result = result.rename(columns={"correction_type": "Ручное решение"})
    rename_map = {
        "image_name": "Изображение",
        "object_id": "Объект",
        "sku_id": "SKU",
        "sku_name": "Название SKU",
        "sku_confidence": "Сходство",
        "distinct_margin": "Отрыв",
        "second_distinct_sku": "Второй SKU",
        "crop_path": "Фрагмент",
        "created_at": "Время",
        "old_status": "Старый статус",
        "old_sku_id": "Старый SKU",
        "new_sku_id": "Новый SKU",
        "comment": "Комментарий",
    }
    return result.rename(columns={key: value for key, value in rename_map.items() if key in result.columns})


def _result_csv(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "04_identification" / "identification_results.csv",
        run_dir / "03_identification" / "identification_results.csv",
        run_dir / "identification_results.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _corrected_results_csv(run_dir: Path) -> Path:
    return run_dir / "04_identification" / "identification_results_corrected.csv"


def _corrections_csv(run_dir: Path) -> Path:
    return run_dir / "manual_corrections.csv"


def _load_corrections(run_dir: Path) -> pd.DataFrame:
    path = _corrections_csv(run_dir)
    if not path.exists():
        return pd.DataFrame(columns=CORRECTION_COLUMNS)
    df = _read_csv(path)
    return df if not df.empty else pd.DataFrame(columns=CORRECTION_COLUMNS)


def _save_correction(run_dir: Path, row: pd.Series, status_col: str, correction_type: str, new_sku_id: str, comment: str) -> Path:
    path = _corrections_csv(run_dir)
    existing = _load_corrections(run_dir)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_name": row.get("image_name", ""),
        "object_id": row.get("object_id", ""),
        "old_status": row.get(status_col, ""),
        "old_sku_id": row.get("sku_id", ""),
        "old_sku_name": row.get("sku_name", ""),
        "new_sku_id": new_sku_id,
        "correction_type": correction_type,
        "comment": comment,
        "sku_confidence": row.get("sku_confidence", ""),
        "distinct_margin": row.get("distinct_margin", ""),
        "second_distinct_sku": row.get("second_distinct_sku", ""),
        "crop_path": row.get("crop_path", ""),
    }
    updated = pd.concat([existing, pd.DataFrame([payload])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated[CORRECTION_COLUMNS].to_csv(path, index=False)
    return path


def _available_sku_values(results_df: pd.DataFrame, gallery_df: pd.DataFrame) -> list[str]:
    values: set[str] = set()
    if "sku_id" in gallery_df.columns:
        values.update(str(x) for x in gallery_df["sku_id"].dropna().unique() if str(x) and str(x) != "nan")
    if "sku_id" in results_df.columns:
        values.update(str(x) for x in results_df["sku_id"].dropna().unique() if str(x) and str(x) != "nan")
    return sorted(values)


def _last_run(output_root: Path) -> Optional[Path]:
    if not output_root.exists():
        return None
    candidates = [p for p in output_root.iterdir() if p.is_dir() and _result_csv(p)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _safe_run_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", raw.strip())
    return cleaned.strip("_") or datetime.now().strftime("analysis_%Y-%m-%d_%H-%M-%S")


def _count_images(images_dir: Path) -> int:
    if not images_dir.exists() or not images_dir.is_dir():
        return 0
    return sum(1 for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _readiness(cfg: Dict[str, Any], images_dir: Path) -> Dict[str, bool]:
    output_root = _to_runtime_path(cfg["output_root"])
    return {
        "Папка изображений": images_dir.exists() and images_dir.is_dir() and _count_images(images_dir) > 0,
        "Предобученная модель": _to_runtime_path(cfg["weights"]).exists(),
        "SKU-галерея": _to_runtime_path(cfg["system_gallery_csv"]).exists() and _to_runtime_path(cfg["system_gallery_dir"]).exists(),
        "Папка результатов": output_root.exists() or output_root.parent.exists(),
    }


def _status_counts(df: pd.DataFrame) -> Dict[str, int]:
    status_col = _status_column(df)
    if df.empty or status_col is None:
        return {"total": 0, "matched": 0, "matched_uncertain": 0, "unknown": 0}
    statuses = df[status_col].astype(str)
    return {
        "total": int(len(df)),
        "matched": int(statuses.eq("matched").sum()),
        "matched_uncertain": int(statuses.eq("matched_uncertain").sum()),
        "unknown": int(statuses.eq("unknown").sum()),
    }


def _metric_row(data: Dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Найдено товаров", data.get("detected_objects", 0))
    c2.metric("Уверенно идентифицировано", data.get("matched", 0))
    c3.metric("Требует проверки", data.get("matched_uncertain", 0))
    c4.metric("Не определено", data.get("unknown", 0))
    c5.metric("С кандидатом", f"{data.get('assigned_share', 0.0) * 100:.2f}%")


def _corrected_metric_row(run_dir: Path) -> None:
    corrected = _read_csv(_corrected_results_csv(run_dir))
    counts = _status_counts(corrected)
    if counts["total"] == 0:
        st.info("Исправленный результат пока не сформирован.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Объектов в исправленной таблице", counts["total"])
    c2.metric("Уверенно после правок", counts["matched"])
    c3.metric("Осталось на проверке", counts["matched_uncertain"])
    c4.metric("Не определено после правок", counts["unknown"])


def _mean_value(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())


def _render_image_grid(paths: list[Path], title: str, limit: int = 9) -> None:
    if not paths:
        st.info(f"{title}: изображения не найдены.")
        return
    st.subheader(title)
    cols = st.columns(3)
    for index, image_path in enumerate(paths[:limit]):
        with cols[index % 3]:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)


def _crop_paths_from_rows(rows: pd.DataFrame, limit: int = 9) -> list[Path]:
    if "crop_path" not in rows.columns:
        return []
    result: list[Path] = []
    for raw in rows["crop_path"].dropna().astype(str).tolist():
        path = _to_runtime_path(raw)
        if path.exists() and path.is_file():
            result.append(path)
        if len(result) >= limit:
            break
    return result


def _sku_status_table(selected: pd.DataFrame, status_col: Optional[str]) -> pd.DataFrame:
    if not status_col:
        return pd.DataFrame(columns=["Статус", "Количество", "Доля"])
    counts = selected[status_col].astype(str).map(_status_label).value_counts().rename_axis("Статус").reset_index(name="Количество")
    counts["Доля"] = counts["Количество"] / max(1, int(counts["Количество"].sum()))
    return counts


def _confusion_table(selected: pd.DataFrame) -> pd.DataFrame:
    if "second_distinct_sku" not in selected.columns:
        return pd.DataFrame()
    subset = selected.copy()
    subset["second_distinct_sku"] = subset["second_distinct_sku"].fillna("").astype(str)
    subset = subset[subset["second_distinct_sku"].ne("")]
    if subset.empty:
        return pd.DataFrame()
    grouped = subset.groupby("second_distinct_sku").agg(
        count=("second_distinct_sku", "size"),
        mean_score=("sku_confidence", "mean") if "sku_confidence" in subset.columns else ("second_distinct_sku", "size"),
        mean_margin=("distinct_margin", "mean") if "distinct_margin" in subset.columns else ("second_distinct_sku", "size"),
    )
    return grouped.sort_values("count", ascending=False).reset_index().rename(columns={"second_distinct_sku": "Конкурирующий SKU"})


def _comparison_table(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    b = _status_counts(before)
    a = _status_counts(after)
    rows = [
        ("Всего объектов", b["total"], a["total"]),
        ("Уверенно идентифицировано", b["matched"], a["matched"]),
        ("Требует проверки", b["matched_uncertain"], a["matched_uncertain"]),
        ("Не определено", b["unknown"], a["unknown"]),
    ]
    return pd.DataFrame(
        [
            {"Показатель": label, "До": before_value, "После": after_value, "Изменение": after_value - before_value}
            for label, before_value, after_value in rows
        ]
    )


def _changed_rows(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before.empty or after.empty or not {"image_name", "object_id"}.issubset(before.columns) or not {"image_name", "object_id"}.issubset(after.columns):
        return pd.DataFrame()
    before_status = _status_column(before)
    after_status = _status_column(after)
    if before_status is None or after_status is None:
        return pd.DataFrame()
    before_part = before[["image_name", "object_id", before_status, "sku_id", "sku_name"]].copy()
    after_part = after[["image_name", "object_id", after_status, "sku_id", "sku_name"]].copy()
    before_part = before_part.rename(columns={before_status: "status_before", "sku_id": "sku_before", "sku_name": "name_before"})
    after_part = after_part.rename(columns={after_status: "status_after", "sku_id": "sku_after", "sku_name": "name_after"})
    merged = before_part.merge(after_part, on=["image_name", "object_id"], how="inner")
    changed = merged[
        merged["status_before"].astype(str).ne(merged["status_after"].astype(str))
        | merged["sku_before"].astype(str).ne(merged["sku_after"].astype(str))
    ]
    if not changed.empty:
        changed["status_before"] = changed["status_before"].map(_status_label)
        changed["status_after"] = changed["status_after"].map(_status_label)
    return changed.rename(
        columns={
            "image_name": "Изображение",
            "object_id": "Объект",
            "status_before": "Статус до",
            "status_after": "Статус после",
            "sku_before": "SKU до",
            "sku_after": "SKU после",
            "name_before": "Название до",
            "name_after": "Название после",
        }
    )


def _download_file(label: str, path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    suffix = path.suffix.lower()
    mime = "text/csv" if suffix == ".csv" else "application/json" if suffix == ".json" else "text/plain"
    try:
        data = path.read_bytes()
    except Exception:
        return
    st.download_button(
        label=f"Скачать: {label}",
        data=data,
        file_name=path.name,
        mime=mime,
        use_container_width=True,
        key=f"download_{label}_{path.name}",
    )


def _next_steps(run_dir: Path) -> None:
    st.subheader("Что делать дальше")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("Проверьте спорные товары и сохраните ручные решения.")
        if st.button("Проверить спорные товары", use_container_width=True, key="next_review"):
            st.session_state["demo_run_dir"] = str(run_dir)
            st.session_state["demo_page"] = "Проверка идентификации"
            st.rerun()
    with c2:
        st.info("Посмотрите, с какими товарами путается конкретный SKU.")
        if st.button("Анализировать SKU", use_container_width=True, key="next_sku"):
            st.session_state["demo_run_dir"] = str(run_dir)
            st.session_state["demo_page"] = "Анализ SKU"
            st.rerun()
    with c3:
        st.info("Сформируйте исправленную таблицу и отчёт по ручным решениям.")
        if st.button("Перейти к отчётам", use_container_width=True, key="next_reports"):
            st.session_state["demo_run_dir"] = str(run_dir)
            st.session_state["demo_page"] = "Отчёты"
            st.rerun()


def _render_result(run_dir: Path, show_table: bool = False) -> None:
    if not run_dir.exists():
        st.warning("Папка результата не найдена.")
        return
    data = collect_summary(run_dir)
    if data["results_df"].empty:
        st.warning("В выбранной папке нет результатов идентификации.")
        return

    validation_summary = data["validation"].get("summary", {}) if isinstance(data["validation"], dict) else {}
    validation_status = validation_summary.get("status", "не проверено")

    st.subheader("Итог анализа")
    st.caption(f"Папка результата: `{run_dir}`")
    st.info(f"Статус проверки результата: **{validation_status}**")
    _metric_row(data)

    corrections = _load_corrections(run_dir)
    if not corrections.empty:
        st.success(f"Сохранено ручных решений: {len(corrections)}")
        _corrected_metric_row(run_dir)

    if data["visualized"]:
        st.subheader("Примеры результата")
        cols = st.columns(3)
        for index, image_path in enumerate(data["visualized"][:9]):
            with cols[index % 3]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)

    _next_steps(run_dir)

    if show_table:
        st.subheader("Таблица SKU-сопоставления")
        st.dataframe(_display_df(data["results_df"]), use_container_width=True, height=420)
        corrected_df = _read_csv(_corrected_results_csv(run_dir))
        if not corrected_df.empty:
            st.subheader("Исправленная таблица с учётом ручных решений")
            st.dataframe(_display_df(corrected_df), use_container_width=True, height=420)


def _build_steps(cfg: Dict[str, Any], images_dir: Path, out_dir: Path) -> list[CommandStep]:
    weights = _to_runtime_path(cfg["weights"])
    gallery_dir = _to_runtime_path(cfg["system_gallery_dir"])
    gallery_csv = _to_runtime_path(cfg["system_gallery_csv"])
    inference_dir = out_dir / "03_query_inference"
    predictions_json = inference_dir / "predictions.json"
    uncertain_dir = out_dir / "uncertain_report"

    inference_cmd = [
        sys.executable,
        "run_inference.py",
        "--model",
        str(cfg["model"]),
        "--weights",
        str(weights),
        "--images-dir",
        str(images_dir),
        "--out-dir",
        str(inference_dir),
        "--conf",
        str(cfg["conf"]),
        "--imgsz",
        str(cfg["imgsz"]),
    ]
    if cfg.get("device"):
        inference_cmd.extend(["--device", str(cfg["device"])])

    identify_cmd = [
        sys.executable,
        "run_existing_photo_identification.py",
        "--out-dir",
        str(out_dir),
        "--gallery-dir",
        str(gallery_dir),
        "--gallery-csv",
        str(gallery_csv),
        "--query-predictions-json",
        str(predictions_json),
        "--threshold",
        str(cfg["threshold"]),
        "--thresholds",
        str(cfg["thresholds"]),
        "--top-k",
        str(cfg["top_k"]),
        "--enable-uncertain-status",
        "--ambiguity-margin",
        str(cfg["ambiguity_margin"]),
        "--visualize-limit",
        str(cfg["visualize_limit"]),
        "--progress-every",
        str(cfg["progress_every"]),
    ]

    return [
        CommandStep("Поиск товаров", inference_cmd, cwd=ROOT, description="Применение предобученной модели к выбранным изображениям."),
        CommandStep("SKU-сопоставление", identify_cmd, cwd=ROOT, description="Сопоставление найденных товаров с системной SKU-галереей."),
        CommandStep("Проверка результата", [sys.executable, "scripts/validate_run_outputs.py", "--run-dir", str(out_dir)], cwd=ROOT),
        CommandStep(
            "Отчёт по спорным товарам",
            [sys.executable, "scripts/build_uncertain_report.py", "--results-csv", str(out_dir / "04_identification" / "identification_results.csv"), "--out-dir", str(uncertain_dir)],
            cwd=ROOT,
        ),
    ]


def page_analysis(cfg: Dict[str, Any]) -> None:
    st.title("ShelfVision")
    st.subheader("Анализ товарных полок")
    st.write("Используется предобученная модель ShelfVision. Выберите папку с изображениями полок и запустите анализ.")

    default_images = str(cfg.get("default_images_dir", ""))
    images_raw = st.text_input("Папка с изображениями", value=default_images)
    run_name = st.text_input("Название запуска, необязательно", value="")

    images_dir = _to_runtime_path(images_raw)
    output_root = _to_runtime_path(cfg["output_root"])
    ready = _readiness(cfg, images_dir)

    st.subheader("Готовность к запуску")
    cols = st.columns(len(ready))
    for index, (label, ok) in enumerate(ready.items()):
        cols[index].metric(label, "готово" if ok else "нужно проверить")

    with st.expander("Системная информация", expanded=False):
        st.write("Активная модель: предобученная модель ShelfVision")
        st.write(f"Путь к модели: `{_display_path(_to_runtime_path(cfg['weights']))}`")
        st.write(f"SKU-галерея: `{_display_path(_to_runtime_path(cfg['system_gallery_csv']))}`")
        st.write(f"Папка результатов: `{_display_path(output_root)}`")
        st.write(f"Рекомендованный режим: порог сопоставления {cfg['threshold']}, проверка спорных случаев включена")
        st.write(f"Найдено изображений в выбранной папке: {_count_images(images_dir)}")

    last = _last_run(output_root)
    if last:
        st.info(f"Найден последний результат: `{last}`")
        if st.button("Показать последний результат", use_container_width=True):
            st.session_state["demo_run_dir"] = str(last)
            st.session_state["demo_page"] = "Результат"
            st.rerun()

    can_run = all(ready.values())
    if st.button("Запустить анализ", type="primary", use_container_width=True, disabled=not can_run):
        output_root.mkdir(parents=True, exist_ok=True)
        out_dir = output_root / _safe_run_name(run_name)
        if out_dir.exists():
            out_dir = output_root / datetime.now().strftime("analysis_%Y-%m-%d_%H-%M-%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = _build_steps(cfg, images_dir, out_dir)
        ok = run_steps_with_progress(steps, title="Анализ полки", success_message="Анализ завершён", failure_message="Анализ завершился с ошибкой")
        if ok:
            st.session_state["demo_run_dir"] = str(out_dir)
            st.success("Анализ завершён. Результат готов к просмотру.")
            _render_result(out_dir)


def page_result(cfg: Dict[str, Any]) -> None:
    st.title("Результат анализа")
    default_run = st.session_state.get("demo_run_dir") or ""
    run_raw = st.text_input("Папка результата", value=default_run)
    if run_raw:
        _render_result(_to_runtime_path(run_raw), show_table=True)
    else:
        st.info("Выберите папку результата или запустите анализ на главном экране.")


def page_review(cfg: Dict[str, Any]) -> None:
    st.title("Проверка идентификации")
    run_raw = st.text_input("Папка результата", value=st.session_state.get("demo_run_dir", ""), key="review_run_dir")
    if not run_raw:
        st.info("Сначала выберите папку результата.")
        return
    run_dir = _to_runtime_path(run_raw)
    data = collect_summary(run_dir)
    results_csv = _result_csv(run_dir)
    df = _read_csv(results_csv)
    status_col = _status_column(df)
    if df.empty or not status_col:
        st.warning("Таблица результатов не найдена или в ней нет статусов.")
        return

    uncertain = df[df[status_col].astype(str).eq("matched_uncertain")].copy()
    corrections = _load_corrections(run_dir)
    st.metric("Товаров, требующих проверки", len(uncertain))
    st.metric("Сохранено ручных решений", len(corrections))
    if uncertain.empty:
        st.success("Спорные товары не найдены.")
        return

    preview_cols = [c for c in ["image_name", "object_id", "sku_id", "sku_name", "best_distinct_sku", "second_distinct_sku", "sku_confidence", "distinct_margin", "crop_path"] if c in uncertain.columns]
    st.dataframe(_display_df(uncertain[preview_cols].head(200)), use_container_width=True, height=320)

    selected_index = st.selectbox("Посмотреть объект", uncertain.index.tolist(), format_func=lambda i: f"строка {i}: объект {uncertain.loc[i].get('object_id', '')}")
    row = uncertain.loc[selected_index]
    c1, c2 = st.columns([1, 2])
    with c1:
        crop_path = row.get("crop_path", "")
        crop = _to_runtime_path(crop_path) if crop_path else None
        if crop and crop.exists():
            st.image(str(crop), caption="Вырезанный фрагмент", use_container_width=True)
        else:
            st.info("Фрагмент товара не найден по указанному пути.")
    with c2:
        st.write(f"Лучший SKU: `{row.get('sku_id', '')}`")
        st.write(f"Название: `{row.get('sku_name', '')}`")
        st.write(f"Сходство: `{row.get('sku_confidence', '')}`")
        st.write(f"Отрыв от второго кандидата: `{row.get('distinct_margin', '')}`")
        st.write(f"Второй SKU: `{row.get('second_distinct_sku', '')}`")
        if "top_k" in row:
            st.code(str(row.get("top_k", "")), language="text")

    st.subheader("Ручное решение")
    sku_options = _available_sku_values(data["results_df"], data["gallery_df"])
    action = st.radio(
        "Что сделать с выбранным объектом",
        ["Подтвердить предложенный SKU", "Выбрать другой SKU", "Оставить не определённым", "Отложить"],
        horizontal=True,
    )
    if action == "Выбрать другой SKU" and sku_options:
        default_sku = str(row.get("sku_id", ""))
        index = sku_options.index(default_sku) if default_sku in sku_options else 0
        new_sku_id = st.selectbox("Правильный SKU", sku_options, index=index)
        correction_type = "change_sku"
    elif action == "Подтвердить предложенный SKU":
        new_sku_id = str(row.get("sku_id", ""))
        correction_type = "confirm_match"
    elif action == "Оставить не определённым":
        new_sku_id = ""
        correction_type = "mark_unknown"
    else:
        new_sku_id = str(row.get("sku_id", ""))
        correction_type = "needs_review"

    comment = st.text_input("Комментарий, необязательно", value="")
    if st.button("Сохранить решение", type="primary", use_container_width=True):
        path = _save_correction(run_dir, row, status_col, correction_type, new_sku_id, comment)
        st.success(f"Решение сохранено: {path}")

    with st.expander("Сохранённые ручные решения", expanded=False):
        updated = _load_corrections(run_dir)
        if updated.empty:
            st.info("Ручные решения пока не сохранены.")
        else:
            st.dataframe(_display_df(updated.tail(100)), use_container_width=True, height=320)


def page_sku(cfg: Dict[str, Any]) -> None:
    st.title("Анализ SKU")
    run_raw = st.text_input("Папка результата", value=st.session_state.get("demo_run_dir", ""), key="sku_run_dir")
    if not run_raw:
        st.info("Сначала выберите папку результата.")
        return
    run_dir = _to_runtime_path(run_raw)
    data = collect_summary(run_dir)
    base_results_df = data["results_df"]
    corrected_df = _read_csv(_corrected_results_csv(run_dir))
    gallery_df = data["gallery_df"]
    if base_results_df.empty:
        st.warning("Результаты SKU-сопоставления не найдены.")
        return

    source_options = ["Исходный результат"]
    if not corrected_df.empty:
        source_options.append("После ручных решений")
    source = st.radio("Источник данных", source_options, horizontal=True)
    results_df = corrected_df if source == "После ручных решений" and not corrected_df.empty else base_results_df

    sku_values = sorted(str(x) for x in results_df.get("sku_id", pd.Series(dtype=str)).dropna().unique() if str(x) and str(x) != "nan")
    if not sku_values:
        st.warning("В результатах нет назначенных SKU.")
        return
    sku = st.selectbox("Выберите SKU", sku_values)
    selected = results_df[results_df["sku_id"].astype(str).eq(sku)].copy()
    status_col = _status_column(selected)

    st.subheader("Сводка по выбранному SKU")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Всего найдено", len(selected))
    if status_col:
        c2.metric("Уверенно", int(selected[status_col].astype(str).eq("matched").sum()))
        c3.metric("Требует проверки", int(selected[status_col].astype(str).eq("matched_uncertain").sum()))
        c4.metric("Не определено", int(selected[status_col].astype(str).eq("unknown").sum()))
    c5.metric("Среднее сходство", f"{_mean_value(selected, 'sku_confidence'):.4f}")

    status_table = _sku_status_table(selected, status_col)
    if not status_table.empty:
        st.subheader("Распределение статусов")
        st.dataframe(status_table, use_container_width=True, height=160)
        st.bar_chart(status_table.set_index("Статус")["Количество"])

    refs = gallery_df[gallery_df.get("sku_id", pd.Series(dtype=str)).astype(str).eq(sku)].head(9) if not gallery_df.empty and "sku_id" in gallery_df.columns else pd.DataFrame()
    if refs.empty:
        st.info("Эталоны выбранного SKU в галерее не найдены.")
    else:
        ref_paths = []
        for _, ref in refs.iterrows():
            image_path = ref.get("image_path", "")
            image = _to_runtime_path(image_path) if image_path else None
            if image and image.exists():
                ref_paths.append(image)
        _render_image_grid(ref_paths, "Эталоны выбранного SKU", limit=9)

    found_paths = _crop_paths_from_rows(selected, limit=9)
    _render_image_grid(found_paths, "Примеры найденных товаров", limit=9)

    if status_col:
        uncertain = selected[selected[status_col].astype(str).eq("matched_uncertain")].copy()
    else:
        uncertain = pd.DataFrame()
    if not uncertain.empty:
        st.subheader("Спорные объекты по выбранному SKU")
        uncertainty_cols = [c for c in ["image_name", "object_id", "sku_id", "sku_name", "second_distinct_sku", "sku_confidence", "distinct_margin", "crop_path"] if c in uncertain.columns]
        st.dataframe(_display_df(uncertain[uncertainty_cols].head(200)), use_container_width=True, height=260)

    confusion = _confusion_table(selected)
    if not confusion.empty:
        st.subheader("С какими SKU чаще всего путается")
        st.dataframe(confusion.head(30), use_container_width=True, height=260)
        st.bar_chart(confusion.head(15).set_index("Конкурирующий SKU")["count"])

    corrections = _load_corrections(run_dir)
    if not corrections.empty:
        related = corrections[
            corrections.get("old_sku_id", pd.Series(dtype=str)).astype(str).eq(sku)
            | corrections.get("new_sku_id", pd.Series(dtype=str)).astype(str).eq(sku)
        ]
        if not related.empty:
            st.subheader("Ручные решения, связанные с этим SKU")
            st.dataframe(_display_df(related.tail(100)), use_container_width=True, height=260)

    st.subheader("Таблица объектов")
    object_cols = [c for c in ["image_name", "object_id", status_col or "", "sku_id", "sku_name", "sku_confidence", "distinct_margin", "second_distinct_sku", "crop_path"] if c and c in selected.columns]
    st.dataframe(_display_df(selected[object_cols].head(300)), use_container_width=True, height=420)


def page_comparison(cfg: Dict[str, Any]) -> None:
    st.title("Сравнение до/после")
    run_raw = st.text_input("Папка результата", value=st.session_state.get("demo_run_dir", ""), key="comparison_run_dir")
    if not run_raw:
        st.info("Сначала выберите папку результата.")
        return
    run_dir = _to_runtime_path(run_raw)
    before = _read_csv(_result_csv(run_dir))
    after = _read_csv(_corrected_results_csv(run_dir))
    corrections = _load_corrections(run_dir)

    if before.empty:
        st.warning("Исходная таблица SKU-сопоставления не найдена.")
        return
    if corrections.empty:
        st.info("Ручные решения пока не сохранены. Сначала откройте раздел 'Проверка идентификации' и сохраните хотя бы одно решение.")
    if after.empty:
        st.info("Исправленная таблица пока не сформирована. Откройте раздел 'Отчёты' и нажмите 'Применить ручные решения'.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Ручных решений", len(corrections))
    c2.metric("Уникальных объектов с решением", len(corrections.drop_duplicates(["image_name", "object_id"])) if not corrections.empty and {"image_name", "object_id"}.issubset(corrections.columns) else 0)
    c3.metric("Исправленная таблица", "готова" if not after.empty else "не готова")

    if not after.empty:
        st.subheader("Изменение итоговых показателей")
        comparison = _comparison_table(before, after)
        st.dataframe(comparison, use_container_width=True, height=220)

        b = _status_counts(before)
        a = _status_counts(after)
        c1, c2, c3 = st.columns(3)
        c1.metric("Уверенно идентифицировано", a["matched"], delta=a["matched"] - b["matched"])
        c2.metric("Требует проверки", a["matched_uncertain"], delta=a["matched_uncertain"] - b["matched_uncertain"], delta_color="inverse")
        c3.metric("Не определено", a["unknown"], delta=a["unknown"] - b["unknown"], delta_color="inverse")

        chart_df = comparison[comparison["Показатель"].ne("Всего объектов")].set_index("Показатель")[["До", "После"]]
        st.bar_chart(chart_df)

        changed = _changed_rows(before, after)
        st.subheader("Какие строки изменились")
        if changed.empty:
            st.info("Изменённые строки не найдены или нет общих ключей image_name/object_id.")
        else:
            st.dataframe(changed.head(300), use_container_width=True, height=360)

    with st.expander("Ручные решения", expanded=False):
        if corrections.empty:
            st.info("Ручные решения не найдены.")
        else:
            st.dataframe(_display_df(corrections.tail(200)), use_container_width=True, height=360)


def page_reports(cfg: Dict[str, Any]) -> None:
    st.title("Отчёты")
    run_raw = st.text_input("Папка результата", value=st.session_state.get("demo_run_dir", ""), key="reports_run_dir")
    if not run_raw:
        st.info("Сначала выберите папку результата.")
        return
    run_dir = _to_runtime_path(run_raw)
    corrected_csv = _corrected_results_csv(run_dir)
    corrections_csv = _corrections_csv(run_dir)
    paths = [
        ("Отчёт проверки", run_dir / "validation_report.md"),
        ("Отчёт по спорным товарам", run_dir / "uncertain_report" / "matched_uncertain_report.md"),
        ("Ручные решения", corrections_csv),
        ("Исправленная таблица", corrected_csv),
        ("Отчёт применения ручных решений", run_dir / "manual_corrections_report.md"),
        ("Таблица SKU-сопоставления", run_dir / "04_identification" / "identification_results.csv"),
        ("Отчёт по связке детекции и идентификации", run_dir / "05_reports" / "segmentation_identification_report.md"),
        ("Итоговые визуализации", run_dir / "04_identification" / "visualized"),
    ]
    for label, path in paths:
        st.write(f"- **{label}:** `{path}` {'✅' if path.exists() else '⚠️'}")

    downloadable = [(label, path) for label, path in paths if path.exists() and path.is_file()]
    if downloadable:
        st.subheader("Скачать материалы")
        cols = st.columns(2)
        for index, (label, path) in enumerate(downloadable):
            with cols[index % 2]:
                _download_file(label, path)

    corrections = _load_corrections(run_dir)
    if not corrections.empty:
        st.subheader("Последние ручные решения")
        st.dataframe(_display_df(corrections.tail(50)), use_container_width=True, height=260)

    if corrections_csv.exists() and st.button("Применить ручные решения", type="primary", use_container_width=True):
        steps = [
            CommandStep(
                "Применение ручных решений",
                [
                    sys.executable,
                    "scripts/apply_manual_corrections.py",
                    "--results-csv",
                    str(run_dir / "04_identification" / "identification_results.csv"),
                    "--corrections-csv",
                    str(corrections_csv),
                    "--out-csv",
                    str(corrected_csv),
                    "--report-dir",
                    str(run_dir),
                ],
                cwd=ROOT,
            )
        ]
        ok = run_steps_with_progress(steps, title="Применение ручных решений", success_message="Ручные решения применены")
        if ok:
            st.success("Исправленная таблица сформирована.")
            _corrected_metric_row(run_dir)

    if st.button("Обновить отчёты проверки", use_container_width=True):
        steps = [
            CommandStep("Проверка результата", [sys.executable, "scripts/validate_run_outputs.py", "--run-dir", str(run_dir)], cwd=ROOT),
            CommandStep("Отчёт по спорным товарам", [sys.executable, "scripts/build_uncertain_report.py", "--results-csv", str(run_dir / "04_identification" / "identification_results.csv"), "--out-dir", str(run_dir / "uncertain_report")], cwd=ROOT),
        ]
        run_steps_with_progress(steps, title="Обновление отчётов", success_message="Отчёты обновлены")


def main() -> None:
    cfg = _demo_config()
    st.set_page_config(page_title=cfg["app_title"], page_icon="🛒", layout="wide")
    pages = ["Анализ полки", "Результат", "Проверка идентификации", "Анализ SKU", "Сравнение до/после", "Отчёты"]
    current = st.sidebar.radio("Раздел", pages, index=pages.index(st.session_state.get("demo_page", "Анализ полки")) if st.session_state.get("demo_page", "Анализ полки") in pages else 0)
    st.session_state["demo_page"] = current

    if current == "Анализ полки":
        page_analysis(cfg)
    elif current == "Результат":
        page_result(cfg)
    elif current == "Проверка идентификации":
        page_review(cfg)
    elif current == "Анализ SKU":
        page_sku(cfg)
    elif current == "Сравнение до/после":
        page_comparison(cfg)
    else:
        page_reports(cfg)


if __name__ == "__main__":
    main()
