from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from path_utils import to_current_os_path
from src.identification.manual_identification_editor import (
    ManualIdentificationEdit,
    append_manual_identification_edit,
    apply_manual_identification_edits,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _p(value: str | Path | None) -> Path:
    return to_current_os_path(value)


def _read_csv(path: Path) -> pd.DataFrame:
    path = _p(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except Exception:
        return pd.DataFrame()


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return 0


def _default_experiment_dir(config: Dict[str, Any]) -> Path:
    full = config.get("full_photo_identification", {})
    return _p(full.get("out_dir") or "D:/1Diplom/shelfvision_results/full_photo_identification")


def _result_options(df: pd.DataFrame, limit: int = 1000) -> List[str]:
    options: List[str] = []
    for idx, row in df.head(limit).iterrows():
        status = str(row.get("sku_status", ""))
        sku = str(row.get("sku_id", "")) or "unknown"
        score = _to_float(row.get("sku_confidence"))
        margin = _to_float(row.get("distinct_margin"))
        image_name = str(row.get("image_name", ""))
        object_id = _to_int(row.get("object_id"))
        options.append(f"{idx}: {image_name} / obj {object_id} | {status} | {sku} | score={score:.3f} | margin={margin:.3f}")
    return options


def _parse_top_k(raw: Any) -> List[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    candidates: List[str] = []
    for chunk in value.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        sku = chunk.split(":", 1)[0].strip()
        if sku and sku not in candidates:
            candidates.append(sku)
    return candidates


def _show_image(path_value: Any, caption: str) -> None:
    path = _p(str(path_value or ""))
    if path.exists() and path.suffix.lower() in IMAGE_EXTS:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Файл не найден: `{path}`")


def _make_edit(row: pd.Series, new_sku_id: str, new_status: str, edit_type: str, comment: str) -> ManualIdentificationEdit:
    old_margin = row.get("distinct_margin")
    return ManualIdentificationEdit(
        edit_id="",
        image_name=str(row.get("image_name", "")),
        image_path=str(row.get("image_path", "")),
        object_id=_to_int(row.get("object_id")),
        crop_path=str(row.get("crop_path", "")),
        old_sku_id=str(row.get("sku_id", "")),
        old_sku_name=str(row.get("sku_name", "")),
        old_status=str(row.get("sku_status", "")),
        old_score=_to_float(row.get("sku_confidence")),
        old_margin=_to_float(old_margin),
        new_sku_id=new_sku_id.strip(),
        new_sku_name=new_sku_id.strip().replace("_", " "),
        new_status=new_status,
        edit_type=edit_type,
        comment=comment,
    )


def page_identification_review(config: Dict[str, Any]) -> None:
    st.subheader("Ручная проверка идентификации")
    st.caption(
        "Страница показывает найденные объекты, top-k кандидатов и позволяет вручную подтвердить, "
        "изменить или отклонить назначение SKU. Исходный CSV не изменяется: правки сохраняются отдельно."
    )

    experiment_dir = _p(
        st.text_input(
            "Папка полного эксперимента",
            value=str(_default_experiment_dir(config)),
            key="ident_review_experiment_dir",
        )
    )
    results_csv = _p(
        st.text_input(
            "CSV результатов идентификации",
            value=str(experiment_dir / "04_identification" / "identification_results.csv"),
            key="ident_review_results_csv",
        )
    )
    edits_csv = _p(
        st.text_input(
            "Журнал ручных правок",
            value=str(experiment_dir / "06_manual_identification" / "manual_identification_edits.csv"),
            key="ident_review_edits_csv",
        )
    )
    corrected_csv = _p(
        st.text_input(
            "Corrected CSV",
            value=str(experiment_dir / "06_manual_identification" / "identification_results_corrected.csv"),
            key="ident_review_corrected_csv",
        )
    )

    df = _read_csv(results_csv)
    if df.empty:
        st.info("Файл результатов идентификации пока не найден или пуст. Сначала запусти полный контур фото-идентификации.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Объектов", len(df))
    c2.metric("matched", int((df.get("sku_status", "") == "matched").sum()))
    c3.metric("matched_uncertain", int((df.get("sku_status", "") == "matched_uncertain").sum()))
    c4.metric("unknown", int((df.get("sku_status", "") == "unknown").sum()))

    st.markdown("#### Фильтры")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        statuses = sorted([str(x) for x in df.get("sku_status", pd.Series(dtype=str)).dropna().unique().tolist()])
        selected_statuses = st.multiselect("Статусы", statuses, default=[x for x in statuses if x in {"matched_uncertain", "unknown"}] or statuses)
    with fc2:
        max_score = st.slider("Максимальная оценка сходства", 0.0, 1.0, 1.0, 0.01)
    with fc3:
        max_margin = st.slider("Максимальный margin", 0.0, 1.0, 1.0, 0.01)
    with fc4:
        image_filter = st.text_input("Фильтр по изображению", value="")

    filtered = df.copy()
    if selected_statuses:
        filtered = filtered[filtered["sku_status"].astype(str).isin(selected_statuses)]
    if "sku_confidence" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["sku_confidence"], errors="coerce").fillna(0.0) <= max_score]
    if "distinct_margin" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["distinct_margin"], errors="coerce").fillna(0.0) <= max_margin]
    if image_filter.strip() and "image_name" in filtered.columns:
        filtered = filtered[filtered["image_name"].astype(str).str.contains(image_filter.strip(), case=False, na=False)]

    st.caption(f"После фильтрации: {len(filtered)} объектов")
    if filtered.empty:
        st.info("По выбранным фильтрам объектов нет.")
        return

    options = _result_options(filtered)
    selected = st.selectbox("Объект для проверки", options, key="ident_review_selected_object")
    row_idx = int(str(selected).split(":", 1)[0])
    row = df.loc[row_idx]

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Проверяемый объект")
        _show_image(row.get("crop_path", ""), "Вырезанный фрагмент")
        st.write(
            {
                "image_name": row.get("image_name", ""),
                "object_id": _to_int(row.get("object_id")),
                "status": row.get("sku_status", ""),
                "sku_id": row.get("sku_id", ""),
                "score": _to_float(row.get("sku_confidence")),
                "margin": _to_float(row.get("distinct_margin")),
            }
        )
    with right:
        st.markdown("#### Кандидаты")
        top_k_candidates = _parse_top_k(row.get("top_k", ""))
        if top_k_candidates:
            st.write("top-k:", ", ".join(top_k_candidates))
        else:
            st.info("В CSV нет поля top_k или оно пустое.")
        st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)

    st.markdown("#### Ручное решение")
    candidate_options = [str(row.get("sku_id", ""))] + top_k_candidates + ["unknown", "new_sku"]
    candidate_options = [x for i, x in enumerate(candidate_options) if x and x not in candidate_options[:i]]
    selected_decision = st.selectbox("Назначение", candidate_options, key="ident_review_decision")
    custom_sku = st.text_input("Другой SKU ID / новый SKU", value="", key="ident_review_custom_sku")
    new_status = st.selectbox("Новый статус", ["matched", "matched_uncertain", "unknown"], index=0, key="ident_review_new_status")
    comment = st.text_input("Комментарий", value="ручная проверка идентификации", key="ident_review_comment")

    btn1, btn2, btn3, btn4 = st.columns(4)
    with btn1:
        if st.button("Подтвердить", use_container_width=True, key="ident_review_confirm"):
            edit = _make_edit(row, str(row.get("sku_id", "")), str(row.get("sku_status", "matched")), "confirm", comment)
            append_manual_identification_edit(edits_csv, edit)
            st.success("Подтверждение добавлено в журнал правок.")
    with btn2:
        if st.button("Изменить SKU", use_container_width=True, key="ident_review_change"):
            sku = custom_sku.strip() or selected_decision
            edit_type = "create_new_sku" if selected_decision == "new_sku" else "change_sku"
            edit = _make_edit(row, sku, new_status, edit_type, comment)
            append_manual_identification_edit(edits_csv, edit)
            st.success(f"Правка добавлена: {sku} / {new_status}")
    with btn3:
        if st.button("Сделать unknown", use_container_width=True, key="ident_review_unknown"):
            edit = _make_edit(row, "", "unknown", "set_unknown", comment)
            append_manual_identification_edit(edits_csv, edit)
            st.success("Правка unknown добавлена.")
    with btn4:
        if st.button("Применить журнал", use_container_width=True, key="ident_review_apply"):
            outputs = apply_manual_identification_edits(
                identification_results_csv=results_csv,
                edits_csv=edits_csv,
                output_csv=corrected_csv,
                report_dir=corrected_csv.parent,
            )
            st.success("Журнал ручных правок применен.")
            for name, path in outputs.items():
                st.write(f"- {name}: `{path}`")

    edits_df = _read_csv(edits_csv)
    if not edits_df.empty:
        st.markdown("#### Журнал ручных правок")
        st.dataframe(edits_df.tail(100), use_container_width=True, hide_index=True)
