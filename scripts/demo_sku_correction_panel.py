from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import streamlit as st

import action_history
from path_utils import to_current_os_path
from src.identification.manual_gallery_editor import (
    EDIT_COLUMNS,
    ManualGalleryEdit,
    append_manual_edit,
    build_manual_gallery_from_edits,
    infer_gallery_dir_from_experiment,
    list_sku_refs,
    read_manual_edits,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
THUMB_WIDTH = 110
TARGET_THUMB_WIDTH = 120


def _p(value: str | Path | None) -> Path:
    return to_current_os_path(value)


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = _p(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except Exception:
        return pd.DataFrame()


def _read_json(path: str | Path) -> Dict[str, Any]:
    path = _p(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _manual_paths(experiment_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    experiment_dir = _p(experiment_dir)
    manual_root = experiment_dir / "06_manual_gallery"
    edits_csv = manual_root / "manual_cluster_edits.csv"
    manual_gallery_dir = manual_root / "sku_gallery_manual"
    manual_gallery_csv = manual_gallery_dir / "gallery.csv"
    manual_identification_dir = manual_root / "manual_identification"
    return manual_root, edits_csv, manual_gallery_dir, manual_gallery_csv, manual_identification_dir


def _source_gallery_dir(experiment_dir: Path, config: Dict[str, Any]) -> Path:
    inferred = infer_gallery_dir_from_experiment(experiment_dir)
    if inferred is not None and _p(inferred).exists():
        return _p(inferred)

    full = config.get("full_photo_identification", {})
    configured = str(full.get("gallery_dir", "")).strip()
    if configured and _p(configured).exists():
        return _p(configured)

    for candidate in [
        _p(experiment_dir) / "02_demo_gallery" / "sku_gallery_final",
        _p(experiment_dir) / "02_demo_gallery",
    ]:
        if candidate.exists():
            return candidate
    return _p(experiment_dir) / "02_demo_gallery"


def _gallery_stats(gallery_dir: Path) -> tuple[Dict[str, List[Path]], int, int]:
    refs_by_sku = list_sku_refs(_p(gallery_dir))
    refs_count = sum(len(refs) for refs in refs_by_sku.values())
    return refs_by_sku, len(refs_by_sku), refs_count


def _write_edits(edits_csv: Path, edits: List[ManualGalleryEdit]) -> None:
    edits_csv = _p(edits_csv)
    edits_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [edit.__dict__ for edit in edits]
    pd.DataFrame(rows, columns=EDIT_COLUMNS).to_csv(edits_csv, index=False)


def _append_event(exp: Path, kind: str, title: str, details: str, artifact: Path | None = None) -> None:
    try:
        action_history.append_event(exp, kind, title, details, artifact)
    except Exception:
        pass


def _backup_existing_dir(path: Path) -> Path | None:
    """Rename an existing output directory before rebuilding it.

    On Windows-mounted drives (`/mnt/d`) a recursive deletion may fail while
    Streamlit still previews images. A rename is faster and avoids the
    `Directory not empty` error during repeated demo runs.
    """

    path = _p(path)
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}_backup_{stamp}")
    index = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}_backup_{stamp}_{index}")
        index += 1
    path.rename(backup)
    return backup


def _read_text(path: Path, limit: int = 25_000) -> str:
    path = _p(path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) > limit:
        return text[:limit] + "\n\n...текст сокращён для отображения..."
    return text


def _show_report(path: Path, title: str) -> None:
    text = _read_text(path)
    if text:
        with st.expander(title, expanded=False):
            st.markdown(text)


def _image_card(path: Path, caption: str, width: int = THUMB_WIDTH) -> None:
    path = _p(path)
    if path.exists() and path.suffix.lower() in IMAGE_EXTS:
        st.image(str(path), caption=caption, width=width)
    else:
        st.caption(caption)


def _display_refs(refs: Iterable[Path], columns: int = 8, limit: int = 12, width: int = THUMB_WIDTH) -> None:
    refs = [ref for ref in list(refs)[:limit] if _p(ref).suffix.lower() in IMAGE_EXTS]
    if not refs:
        st.info("Эталонные изображения не найдены.")
        return
    cols = st.columns(max(1, min(columns, len(refs))))
    for index, ref in enumerate(refs):
        with cols[index % len(cols)]:
            _image_card(_p(ref), _p(ref).name, width=width)


def _select_refs_grid(refs: List[Path], key_prefix: str, columns: int = 8) -> List[str]:
    if not refs:
        st.info("В выбранном SKU нет эталонных изображений.")
        return []

    selected: List[str] = []
    cols = st.columns(max(1, min(columns, len(refs))))
    for index, ref in enumerate(refs):
        ref = _p(ref)
        with cols[index % len(cols)]:
            _image_card(ref, ref.name, width=THUMB_WIDTH)
            if st.checkbox("вынести", key=f"{key_prefix}_{index}_{ref.name}"):
                selected.append(ref.name)
    return selected


def _iter_focus_candidates(
    refs_by_sku: Dict[str, List[Path]],
    target_sku: str,
    source_filter: str,
    limit: int,
) -> List[tuple[str, Path]]:
    rows: List[tuple[str, Path]] = []
    source_filter = source_filter.strip().lower()
    for source_sku, refs in sorted(refs_by_sku.items()):
        if source_sku == target_sku:
            continue
        if source_filter and source_filter not in source_sku.lower():
            continue
        for ref in refs:
            rows.append((source_sku, _p(ref)))
            if len(rows) >= limit:
                return rows
    return rows


def _render_focus_sku(exp: Path, edits_csv: Path, refs_by_sku: Dict[str, List[Path]]) -> None:
    st.markdown("#### Улучшить идентификацию конкретного SKU")
    st.caption(
        "Выберите целевой SKU, затем отметьте эталоны из других автоматически созданных SKU, "
        "которые должны относиться к нему. При применении они будут перенесены в целевой SKU."
    )

    sku_ids = sorted(refs_by_sku.keys())
    if not sku_ids:
        st.info("SKU-галерея пуста.")
        return

    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        target_sku = st.selectbox(
            "Целевой SKU",
            sku_ids,
            format_func=lambda sku: f"{sku} ({len(refs_by_sku.get(sku, []))} эталонов)",
            key="demo_focus_target_sku",
        )
    with c2:
        source_filter = st.text_input(
            "Фильтр по SKU-источнику",
            value="",
            placeholder="например: 041",
            key="demo_focus_source_filter",
        )
    with c3:
        limit = st.number_input(
            "Сколько кандидатов показать",
            min_value=12,
            max_value=240,
            value=48,
            step=12,
            key="demo_focus_candidate_limit",
        )

    st.markdown(f"##### Текущие эталоны целевого SKU `{target_sku}`")
    _display_refs(refs_by_sku.get(target_sku, []), columns=10, limit=20, width=TARGET_THUMB_WIDTH)

    candidates = _iter_focus_candidates(refs_by_sku, target_sku, source_filter, int(limit))
    if not candidates:
        st.info("Кандидаты из других SKU не найдены. Ослабьте фильтр.")
        return

    st.markdown("##### Отметьте фрагменты, которые должны относиться к целевому SKU")
    selected: Dict[str, List[str]] = {}
    cols = st.columns(10)
    for index, (source_sku, ref) in enumerate(candidates):
        with cols[index % len(cols)]:
            _image_card(ref, f"{source_sku}\n{ref.name}", width=THUMB_WIDTH)
            if st.checkbox("к целевому", key=f"demo_focus_pick_{source_sku}_{index}_{ref.name}"):
                selected.setdefault(source_sku, []).append(ref.name)

    selected_count = sum(len(items) for items in selected.values())
    comment = st.text_input(
        "Комментарий",
        value=f"перенос эталонов в целевой SKU {target_sku}",
        key="demo_focus_comment",
    )
    if st.button(
        f"Перенести выбранные эталоны в `{target_sku}`",
        type="primary",
        use_container_width=True,
        disabled=selected_count == 0,
    ):
        for source_sku, ref_names in selected.items():
            append_manual_edit(
                edits_csv,
                ManualGalleryEdit(
                    operation="split",
                    source_sku_id=source_sku,
                    new_sku_id=target_sku,
                    ref_files=";".join(ref_names),
                    comment=comment,
                ),
            )
        _append_event(
            exp,
            "manual_gallery_focus_sku",
            "Добавлен перенос эталонов в целевой SKU",
            f"target={target_sku}; refs={selected_count}",
            edits_csv,
        )
        st.success(f"Добавлено операций: {len(selected)}; перенесено эталонов: {selected_count}")
        st.rerun()


def _load_candidates(experiment_dir: Path) -> pd.DataFrame:
    candidates = _read_csv(_p(experiment_dir) / "07_sku_audit" / "merge_candidates.csv")
    required = {"sku_a", "sku_b"}
    if candidates.empty or not required.issubset(set(candidates.columns)):
        return pd.DataFrame()
    return candidates


def _candidate_label(row: pd.Series, index: int) -> str:
    sku_a = str(row.get("sku_a", ""))
    sku_b = str(row.get("sku_b", ""))
    similarity = row.get("centroid_similarity", row.get("similarity", ""))
    try:
        suffix = f" | сходство={float(similarity):.3f}"
    except Exception:
        suffix = ""
    return f"{index}: {sku_a} ↔ {sku_b}{suffix}"


def _render_pair_comparison(
    exp: Path,
    edits_csv: Path,
    refs_by_sku: Dict[str, List[Path]],
) -> None:
    st.markdown("#### Похожие SKU")
    st.caption("Проверьте пару визуально похожих SKU и добавьте операцию объединения.")

    candidates = _load_candidates(exp)
    sku_ids = sorted(refs_by_sku.keys())

    if not candidates.empty:
        st.success(f"Найдено кандидатов на объединение: {len(candidates)}")
        options = [_candidate_label(row, idx) for idx, row in candidates.head(300).iterrows()]
        selected = st.selectbox("Кандидат на объединение", options, key="demo_sku_merge_candidate")
        idx = int(str(selected).split(":", 1)[0])
        row = candidates.loc[idx]
        sku_a = str(row.get("sku_a", ""))
        sku_b = str(row.get("sku_b", ""))
        sheet = _p(str(row.get("pair_contact_sheet", "")))
        if sheet.exists():
            st.image(str(sheet), caption=sheet.name, use_container_width=True)
    else:
        st.info("Файл 07_sku_audit/merge_candidates.csv не найден. Пару можно выбрать вручную.")
        if len(sku_ids) < 2:
            st.warning("В галерее меньше двух SKU, объединение недоступно.")
            return
        c1, c2 = st.columns(2)
        with c1:
            sku_a = st.selectbox("SKU A", sku_ids, key="demo_sku_manual_merge_a")
        with c2:
            sku_b = st.selectbox("SKU B", [sku for sku in sku_ids if sku != sku_a], key="demo_sku_manual_merge_b")

    if not sku_a or not sku_b or sku_a == sku_b:
        st.warning("Выберите два разных SKU.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"##### SKU A: `{sku_a}`")
        _display_refs(refs_by_sku.get(sku_a, []), columns=5, limit=10)
    with col_b:
        st.markdown(f"##### SKU B: `{sku_b}`")
        _display_refs(refs_by_sku.get(sku_b, []), columns=5, limit=10)

    comment = st.text_input(
        "Комментарий к объединению",
        value=f"визуальная проверка пары {sku_a} / {sku_b}",
        key="demo_sku_merge_comment",
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button(f"Оставить {sku_a}: объединить {sku_b} → {sku_a}", use_container_width=True):
            append_manual_edit(
                edits_csv,
                ManualGalleryEdit(operation="merge", source_sku_id=sku_b, target_sku_id=sku_a, comment=comment),
            )
            _append_event(exp, "manual_gallery_merge", "Добавлено объединение SKU", f"{sku_b} -> {sku_a}", edits_csv)
            st.success(f"Операция добавлена: {sku_b} -> {sku_a}")
            st.rerun()
    with b2:
        if st.button(f"Оставить {sku_b}: объединить {sku_a} → {sku_b}", use_container_width=True):
            append_manual_edit(
                edits_csv,
                ManualGalleryEdit(operation="merge", source_sku_id=sku_a, target_sku_id=sku_b, comment=comment),
            )
            _append_event(exp, "manual_gallery_merge", "Добавлено объединение SKU", f"{sku_a} -> {sku_b}", edits_csv)
            st.success(f"Операция добавлена: {sku_a} -> {sku_b}")
            st.rerun()
    with b3:
        st.caption("Если это разные товары, операцию добавлять не нужно.")


def _render_split(exp: Path, edits_csv: Path, refs_by_sku: Dict[str, List[Path]]) -> None:
    st.markdown("#### Разделить смешанный SKU")
    st.caption("Выберите эталоны, которые ошибочно попали в текущий SKU. Они будут вынесены в новый SKU.")

    sku_ids = sorted(refs_by_sku.keys())
    if not sku_ids:
        st.info("SKU-галерея пуста.")
        return

    source_sku = st.selectbox(
        "SKU для проверки",
        sku_ids,
        format_func=lambda sku: f"{sku} ({len(refs_by_sku.get(sku, []))} эталонов)",
        key="demo_sku_split_source",
    )
    selected_refs = _select_refs_grid(refs_by_sku.get(source_sku, []), key_prefix=f"demo_sku_split_{source_sku}")

    c1, c2 = st.columns([1, 2])
    with c1:
        new_sku_id = st.text_input("Новый SKU ID", value="", placeholder="пусто = автоматически", key="demo_sku_split_new_sku")
    with c2:
        comment = st.text_input(
            "Комментарий к разделению",
            value="другой товар внутри автоматически созданного SKU",
            key="demo_sku_split_comment",
        )

    if st.button(
        "Вынести выбранные эталоны в новый SKU",
        type="primary",
        use_container_width=True,
        disabled=not selected_refs,
    ):
        append_manual_edit(
            edits_csv,
            ManualGalleryEdit(
                operation="split",
                source_sku_id=source_sku,
                new_sku_id=new_sku_id.strip(),
                ref_files=";".join(selected_refs),
                comment=comment,
            ),
        )
        _append_event(exp, "manual_gallery_split", "Добавлено разделение SKU", f"{source_sku}: {', '.join(selected_refs)}", edits_csv)
        st.success(f"Операция разделения добавлена: {source_sku}")
        st.rerun()


def _render_journal(exp: Path, edits_csv: Path) -> None:
    st.markdown("#### Журнал операций")
    edits = read_manual_edits(edits_csv)
    if not edits:
        st.info(f"Ручных операций пока нет: `{edits_csv}`")
    else:
        st.dataframe(pd.DataFrame([edit.__dict__ for edit in edits], columns=EDIT_COLUMNS), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Удалить последнюю операцию", use_container_width=True, disabled=not edits):
            _write_edits(edits_csv, edits[:-1])
            _append_event(exp, "manual_gallery_undo", "Удалена последняя операция SKU-галереи", str(edits_csv), edits_csv)
            st.rerun()
    with c2:
        if st.button("Очистить журнал", use_container_width=True, disabled=not edits):
            _write_edits(edits_csv, [])
            _append_event(exp, "manual_gallery_clear", "Очищен журнал операций SKU-галереи", str(edits_csv), edits_csv)
            st.rerun()
    with c3:
        edits_csv = _p(edits_csv)
        if edits_csv.exists():
            st.download_button(
                "Скачать CSV журнала",
                data=edits_csv.read_bytes(),
                file_name=edits_csv.name,
                mime="text/csv",
                use_container_width=True,
            )


def _run_manual_identification(
    exp: Path,
    config: Dict[str, Any],
    manual_gallery_dir: Path,
    manual_gallery_csv: Path,
    manual_identification_dir: Path,
) -> subprocess.CompletedProcess[str]:
    full = config.get("full_photo_identification", {})
    query_predictions = _p(exp) / "03_query_inference" / "predictions.json"
    cmd = [
        sys.executable,
        str(ROOT / "run_existing_photo_identification.py"),
        "--out-dir",
        str(_p(manual_identification_dir)),
        "--query-predictions-json",
        str(query_predictions),
        "--gallery-dir",
        str(_p(manual_gallery_dir)),
        "--gallery-csv",
        str(_p(manual_gallery_csv)),
        "--threshold",
        str(full.get("threshold", 0.65)),
        "--thresholds",
        str(full.get("thresholds", "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")),
        "--top-k",
        str(full.get("top_k", 5)),
        "--ambiguity-margin",
        str(full.get("ambiguity_margin", 0.03)),
        "--visualize-limit",
        "60",
        "--progress-every",
        "25",
    ]
    if bool(full.get("enable_uncertain_status", True)):
        cmd.append("--enable-uncertain-status")
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)


def _render_apply(exp: Path, config: Dict[str, Any], source_gallery_dir: Path, edits_csv: Path) -> None:
    manual_root, _, manual_gallery_dir, manual_gallery_csv, manual_identification_dir = _manual_paths(exp)
    st.markdown("#### Применить коррекцию")
    st.write(f"Журнал операций: `{edits_csv}`")
    st.write(f"Ручная галерея: `{manual_gallery_dir}`")
    st.write(f"CSV ручной галереи: `{manual_gallery_csv}`")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("1. Собрать ручную SKU-галерею", type="primary", use_container_width=True):
            try:
                backup_dir = _backup_existing_dir(manual_gallery_dir)
                outputs = build_manual_gallery_from_edits(
                    source_gallery_dir=source_gallery_dir,
                    output_gallery_dir=manual_gallery_dir,
                    edits_csv=edits_csv,
                    out_dir=manual_root,
                    output_gallery_csv=manual_gallery_csv,
                )
                if backup_dir is not None:
                    st.caption(f"Предыдущая ручная галерея сохранена: `{backup_dir}`")
                _append_event(exp, "manual_gallery_build", "Собрана ручная SKU-галерея", str(manual_gallery_csv), outputs.get("summary_json"))
                st.success("Ручная SKU-галерея собрана.")
                for name, path in outputs.items():
                    st.caption(f"{name}: `{_p(path)}`")
            except Exception as exc:
                st.error(f"Ошибка сборки ручной галереи: {exc}")
    with c2:
        if st.button(
            "2. Пересчитать идентификацию с ручной галереей",
            use_container_width=True,
            disabled=not _p(manual_gallery_csv).exists(),
        ):
            with st.spinner("Пересчёт идентификации по готовым query-предсказаниям..."):
                result = _run_manual_identification(exp, config, manual_gallery_dir, manual_gallery_csv, manual_identification_dir)
            if result.returncode == 0:
                summary_path = manual_identification_dir / "05_reports" / "existing_identification_summary.json"
                _append_event(exp, "manual_gallery_identification", "Пересчитана идентификация с ручной галереей", str(manual_identification_dir), summary_path)
                st.success("Идентификация с ручной галереей пересчитана.")
            else:
                st.error("Пересчёт завершился ошибкой.")
            with st.expander("Вывод команды", expanded=result.returncode != 0):
                st.code((result.stdout or "") + "\n" + (result.stderr or ""), language="text")

    _show_report(manual_root / "manual_gallery_report.md", "Отчёт по ручной SKU-галерее")
    if (manual_root / "manual_gallery_summary.json").exists():
        with st.expander("Сводка ручной SKU-галереи", expanded=False):
            st.json(_read_json(manual_root / "manual_gallery_summary.json"))


def _extract_metrics(raw: Dict[str, Any]) -> Dict[str, float | int]:
    if not raw:
        return {}
    return {
        "objects": int(raw.get("query_objects_count", raw.get("total_objects", 0)) or 0),
        "matched": int(raw.get("matched", 0) or 0),
        "matched_uncertain": int(raw.get("matched_uncertain", 0) or 0),
        "unknown": int(raw.get("unknown", 0) or 0),
        "assigned_rate": float(raw.get("assigned_rate", 0.0) or 0.0),
        "matched_rate": float(raw.get("matched_rate", 0.0) or 0.0),
        "unknown_rate": float(raw.get("unknown_rate", 0.0) or 0.0),
        "avg_similarity": float(raw.get("avg_similarity", 0.0) or 0.0),
    }


def _original_metrics(exp: Path) -> tuple[Path | None, Dict[str, float | int]]:
    for path in [
        _p(exp) / "05_reports" / "full_experiment_summary.json",
        _p(exp) / "05_reports" / "existing_identification_summary.json",
        _p(exp) / "04_identification" / "identification_metrics.json",
    ]:
        metrics = _extract_metrics(_read_json(path))
        if metrics:
            return path, metrics
    return None, {}


def _manual_metrics(exp: Path) -> tuple[Path | None, Dict[str, float | int]]:
    manual_root, _, _, _, manual_identification_dir = _manual_paths(exp)
    for path in [
        manual_identification_dir / "05_reports" / "existing_identification_summary.json",
        manual_identification_dir / "04_identification" / "identification_metrics.json",
        manual_root / "manual_gallery_summary.json",
    ]:
        metrics = _extract_metrics(_read_json(path))
        if metrics:
            return path, metrics
    return None, {}


def _metric_card(title: str, metrics: Dict[str, float | int], source: Path | None) -> None:
    st.markdown(f"##### {title}")
    if not metrics:
        st.info("Метрики пока не найдены.")
        return
    st.metric("Доля с кандидатом", f"{float(metrics.get('assigned_rate', 0.0)):.4f}")
    st.write(f"Объектов: `{metrics.get('objects', 0)}`")
    st.write(f"Уверенные: `{metrics.get('matched', 0)}`")
    st.write(f"Требуют проверки: `{metrics.get('matched_uncertain', 0)}`")
    st.write(f"Не определены: `{metrics.get('unknown', 0)}`")
    st.write(f"Среднее сходство: `{float(metrics.get('avg_similarity', 0.0)):.4f}`")
    if source:
        st.caption(f"Источник: `{source}`")


def _render_before_after(exp: Path) -> None:
    st.markdown("#### До/после коррекции SKU-галереи")
    original_path, original = _original_metrics(exp)
    manual_path, manual = _manual_metrics(exp)

    c1, c2 = st.columns(2)
    with c1:
        _metric_card("Исходная галерея", original, original_path)
    with c2:
        _metric_card("Ручная галерея", manual, manual_path)

    if not original or not manual:
        st.info("Для сравнения сначала соберите ручную галерею и пересчитайте идентификацию.")
        return

    rows = []
    labels = {
        "objects": "Объекты query",
        "matched": "Уверенные совпадения",
        "matched_uncertain": "Неоднозначные совпадения",
        "unknown": "Неопределённые объекты",
        "assigned_rate": "Доля с кандидатом",
        "avg_similarity": "Среднее визуальное сходство",
    }
    for key, label in labels.items():
        before = original.get(key, 0)
        after = manual.get(key, 0)
        rows.append({"метрика": label, "до": before, "после": after, "изменение": after - before})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_demo_sku_correction(config: Dict[str, Any], experiment_dir: str | Path) -> None:
    exp = _p(experiment_dir)
    st.subheader("Коррекция SKU-галереи")
    st.caption(
        "Визуальное объединение похожих SKU и разделение смешанных SKU-кластеров. "
        "Исходные результаты эксперимента не изменяются."
    )

    if not exp.exists():
        st.info("Сначала укажите существующую папку результатов в боковой панели.")
        return

    source_gallery_dir = _source_gallery_dir(exp, config)
    if not source_gallery_dir.exists():
        st.warning(f"Исходная SKU-галерея не найдена: `{source_gallery_dir}`")
        return

    manual_root, edits_csv, manual_gallery_dir, manual_gallery_csv, _ = _manual_paths(exp)
    refs_by_sku, sku_count, refs_count = _gallery_stats(source_gallery_dir)
    edits = read_manual_edits(edits_csv)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKU в галерее", sku_count)
    c2.metric("Эталонов", refs_count)
    c3.metric("Ручных операций", len(edits))
    c4.metric("Ручная галерея", "готова" if manual_gallery_csv.exists() else "нет")

    st.info(
        "Используйте этот экран для демонстрации экспертной корректировки: "
        "объедините два SKU, если это один товар, или вынесите ошибочные эталоны в новый SKU."
    )
    st.caption(f"Исходная галерея: `{source_gallery_dir}`")
    st.caption(f"Журнал операций: `{edits_csv}`")

    tabs = st.tabs(["Улучшить SKU", "Похожие SKU", "Разделить SKU", "Журнал", "Применить", "До/после"])
    with tabs[0]:
        _render_focus_sku(exp, edits_csv, refs_by_sku)
    with tabs[1]:
        _render_pair_comparison(exp, edits_csv, refs_by_sku)
    with tabs[2]:
        _render_split(exp, edits_csv, refs_by_sku)
    with tabs[3]:
        _render_journal(exp, edits_csv)
    with tabs[4]:
        _render_apply(exp, config, source_gallery_dir, edits_csv)
    with tabs[5]:
        _render_before_after(exp)
