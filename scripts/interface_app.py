from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


# -----------------------------------------------------------------------------
# Визуальный интерфейс ShelfVision
# -----------------------------------------------------------------------------

APP_TITLE = "ShelfVision: интерфейс экспериментов"
CSV_FILES = {
    "DIR1_models": "reports/all_stats/DIR1_models.csv",
    "YOLO_11_ablations": "reports/all_stats/YOLO_11_ablations.csv",
    "DIR3_WBF": "reports/all_stats/DIR3_WBF.csv",
    "DIR5_robustness": "reports/all_stats/DIR5_robustness.csv",
    "D2S_YOLO_SEG_last": "reports/all_stats/D2S_YOLO_SEG_last.csv",
    "OVERALL_detection_min": "reports/all_stats/OVERALL_detection_min.csv",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

COLUMN_LABELS_RU = {
    "model": "Модель",
    "system": "Система",
    "name": "Название",
    "exp": "Эксперимент",
    "mode": "Режим",
    "is_best": "Лучшая конфигурация",
    "AP": "AP",
    "AP50": "AP50",
    "AP75": "AP75",
    "AP50-95": "AP50-95",
    "AR100": "AR100",
    "P": "Точность",
    "R": "Полнота",
    "mAP50-95": "mAP50-95",
    "minutes_total": "Время обучения, мин",
    "ms_per_image": "Время на изображение, мс",
    "time_ms": "Время, мс",
    "AP_delta": "Изменение AP",
    "AP_delta_%": "Изменение AP, %",
    "P_box": "Точность bbox",
    "R_box": "Полнота bbox",
    "mAP5095_box": "mAP50-95 bbox",
    "mAP5095_mask": "mAP50-95 mask",
    "mAP50_box": "mAP50 bbox",
    "mAP50_mask": "mAP50 mask",
    "mAP5095_mask": "mAP50-95 mask",
}


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_LABELS_RU)


# -----------------------------------------------------------------------------
# Вспомогательные функции
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def existing_csvs(root: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for name, rel in CSV_FILES.items():
        path = root / rel
        if path.exists():
            result[name] = path
    return result


def load_table(root: Path, name: str) -> Optional[pd.DataFrame]:
    path = root / CSV_FILES[name]
    if not path.exists():
        return None
    try:
        return read_csv_cached(str(path))
    except Exception as exc:
        st.error(f"Не удалось прочитать {path}: {exc}")
        return None


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def pick_metric_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def show_dataframe(df: pd.DataFrame, height: int = 420) -> None:
    st.dataframe(_display_df(df), use_container_width=True, height=height)


def metric_card(label: str, value: object, help_text: str = "") -> None:
    st.metric(label=label, value="-" if value is None else value, help=help_text or None)


def format_float(value: object, ndigits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def find_images(root: Path, max_images: int = 200) -> List[Path]:
    preferred_dirs = [
        root / "artifacts" / "final_showcase" / "best_demo",
        root / "artifacts" / "final_showcase" / "all_models",
        root / "reports" / "course_project" / "plots",
        root / "reports" / "master" / "plots",
        root / "artifacts",
        root / "reports",
    ]

    images: List[Path] = []
    seen = set()
    for folder in preferred_dirs:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTS and path.is_file():
                key = str(path.resolve())
                if key not in seen:
                    images.append(path)
                    seen.add(key)
            if len(images) >= max_images:
                return images
    return images


def run_command(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        return proc.returncode, proc.stdout
    except Exception as exc:
        return 1, str(exc)


def package_status() -> pd.DataFrame:
    packages = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "opencv-python",
        "ultralytics",
        "detectron2",
        "torch",
        "ensemble-boxes",
        "psutil",
    ]
    rows = []
    for pkg in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            version = ""
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith("Version:"):
                        version = line.split(":", 1)[1].strip()
                        break
            rows.append({"Компонент": pkg, "Статус": "найден" if result.returncode == 0 else "не найден", "Версия": version})
        except Exception:
            rows.append({"Компонент": pkg, "Статус": "ошибка проверки", "Версия": ""})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Боковая панель
# -----------------------------------------------------------------------------

def render_sidebar(root: Path) -> None:
    st.sidebar.title("ShelfVision")
    st.sidebar.caption("Визуальный интерфейс экспериментов")

    st.sidebar.markdown("### Статус данных")
    csvs = existing_csvs(root)
    for name in CSV_FILES:
        ok = name in csvs
        st.sidebar.write(("✅" if ok else "❌") + f" {name}")

    st.sidebar.markdown("### Быстрый запуск")
    st.sidebar.code("streamlit run scripts/interface_app.py", language="bash")
    st.sidebar.caption(f"Корень проекта: {root}")

    if px is None:
        st.sidebar.warning("Plotly не установлен: графики скрыты. Запусти start_control_panel.bat ещё раз или установи: pip install plotly")


# -----------------------------------------------------------------------------
# Страницы
# -----------------------------------------------------------------------------

def page_home(root: Path) -> None:
    st.header("Главная панель")
    st.write(
        "Интерфейс показывает, какие экспериментальные материалы есть в проекте, "
        "какие модели сравнивались и какие результаты используются в отчётах по проекту и ВКР."
    )

    if px is None:
        st.warning(
            "Plotly не установлен в текущем Python-окружении, поэтому интерактивные графики не отображаются. "
            "Если интерфейс открыт из основной панели, перезапусти `start_control_panel.bat`: он доустановит `plotly`. "
            "Или выполни вручную: `.venv\\Scripts\\python.exe -m pip install plotly`."
        )

    csvs = existing_csvs(root)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("CSV-файлов найдено", f"{len(csvs)}/{len(CSV_FILES)}")
    with c2:
        img_count = len(find_images(root, max_images=500))
        metric_card("Изображений найдено", img_count)
    with c3:
        overall = load_table(root, "OVERALL_detection_min")
        best_ap = None
        if overall is not None and "AP50-95" in overall.columns:
            best_ap = to_number(overall["AP50-95"]).max()
        metric_card("Лучший AP50-95", format_float(best_ap))
    with c4:
        yolo = load_table(root, "YOLO_11_ablations")
        best_name = "-"
        if yolo is not None and "is_best" in yolo.columns and "exp" in yolo.columns:
            best_rows = yolo[yolo["is_best"].astype(str).str.lower() == "true"]
            if not best_rows.empty:
                best_name = str(best_rows.iloc[0]["exp"])
        metric_card("Лучший YOLO", best_name)

    st.subheader("Конвейер эксперимента")
    st.markdown(
        """
        1. Изображения и аннотации SKU110K загружаются в проект.
        2. Модели строят предсказания: рамки, confidence и дополнительные значения.
        3. Предсказания сравниваются с эталонными аннотациями.
        4. Считаются AP50-95, AP50, AP75, точность, полнота, AR100 и скорость.
        5. Результаты сохраняются в CSV и JSON.
        6. Интерфейс показывает таблицы, графики и визуальные примеры.
        """
    )

    st.subheader("Файлы экспериментов")
    rows = []
    for name, rel in CSV_FILES.items():
        path = root / rel
        rows.append({
            "Эксперимент": name,
            "Путь": rel,
            "Найден": "да" if path.exists() else "нет",
            "Размер, КБ": round(path.stat().st_size / 1024, 2) if path.exists() else None,
        })
    show_dataframe(pd.DataFrame(rows), height=260)


def page_detection(root: Path) -> None:
    st.header("Сравнение моделей детекции")
    df = load_table(root, "DIR1_models")
    if df is None:
        st.warning("Файл DIR1_models.csv не найден.")
        return

    show_dataframe(df)

    ap_col = pick_metric_column(df, "AP50-95", "AP")
    model_col = pick_metric_column(df, "model", "system", "name")
    if px and ap_col and model_col:
        chart_df = df.copy()
        chart_df[ap_col] = to_number(chart_df[ap_col])
        fig = px.bar(chart_df, x=model_col, y=ap_col, title="Сравнение моделей по AP50-95", text=ap_col)
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    speed_col = pick_metric_column(df, "ms_per_image", "time_ms", "Время, мс")
    if px and ap_col and speed_col and model_col:
        chart_df = df.copy()
        chart_df[ap_col] = to_number(chart_df[ap_col])
        chart_df[speed_col] = to_number(chart_df[speed_col])
        chart_df = chart_df.dropna(subset=[ap_col, speed_col])
        if not chart_df.empty:
            fig = px.scatter(
                chart_df,
                x=speed_col,
                y=ap_col,
                text=model_col,
                size=ap_col,
                title="Качество и скорость моделей",
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Этот блок нужен для защиты: он показывает, что были сравнены разные подходы "
        "к детекции товаров, а выбор модели основан на метриках, а не на предположении."
    )


def page_yolo_ablations(root: Path) -> None:
    st.header("Абляционный анализ YOLO")
    df = load_table(root, "YOLO_11_ablations")
    if df is None:
        st.warning("Файл YOLO_11_ablations.csv не найден.")
        return

    show_dataframe(df)

    if "is_best" in df.columns:
        best = df[df["is_best"].astype(str).str.lower() == "true"]
        if not best.empty:
            st.success("Лучшая конфигурация: " + str(best.iloc[0].get("exp", "-")))
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Точность", format_float(best.iloc[0].get("P")))
            with c2:
                metric_card("Полнота", format_float(best.iloc[0].get("R")))
            with c3:
                metric_card("mAP50-95", format_float(best.iloc[0].get("mAP50-95")))
            with c4:
                metric_card("Минут", format_float(best.iloc[0].get("minutes_total"), 2))

    if px and "exp" in df.columns and "mAP50-95" in df.columns:
        chart_df = df.copy()
        chart_df["mAP50-95"] = to_number(chart_df["mAP50-95"])
        fig = px.bar(chart_df, x="exp", y="mAP50-95", title="YOLO-абляции по mAP50-95")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    if px and "minutes_total" in df.columns and "mAP50-95" in df.columns:
        chart_df = df.copy()
        chart_df["minutes_total"] = to_number(chart_df["minutes_total"])
        chart_df["mAP50-95"] = to_number(chart_df["mAP50-95"])
        fig = px.scatter(
            chart_df,
            x="minutes_total",
            y="mAP50-95",
            text="exp" if "exp" in chart_df.columns else None,
            title="Баланс качества и времени обучения YOLO",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Этот блок показывает, что была проверена не одна модель, а несколько настроек YOLO. "
        "Лучшей стала конфигурация E03_imgsz_640."
    )


def page_wbf(root: Path) -> None:
    st.header("WBF-ансамбль")
    df = load_table(root, "DIR3_WBF")
    if df is None:
        st.warning("Файл DIR3_WBF.csv не найден.")
        return

    show_dataframe(df)

    if px and "system" in df.columns:
        metrics = [c for c in ["AP", "AP50", "AP75", "AR100"] if c in df.columns]
        if metrics:
            long = df.melt(id_vars=["system"], value_vars=metrics, var_name="Метрика", value_name="Значение")
            long["Значение"] = to_number(long["Значение"])
            fig = px.bar(long, x="system", y="Значение", color="Метрика", barmode="group", title="YOLO, RT-DETR и WBF")
            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Этот блок нужен для пояснения: WBF повышает полноту AR100, но не дал лучшего AP50-95. "
        "Поэтому он оставлен как дополнительный эксперимент."
    )


def page_robustness(root: Path) -> None:
    st.header("Устойчивость к искажениям")
    df = load_table(root, "DIR5_robustness")
    if df is None:
        st.warning("Файл DIR5_robustness.csv не найден.")
        return

    df = df.copy()
    show_dataframe(df)

    if "AP" in df.columns and "mode" in df.columns:
        df["AP"] = to_number(df["AP"])
        clean_rows = df[df["mode"].astype(str) == "clean"]
        if not clean_rows.empty:
            clean_ap = float(clean_rows.iloc[0]["AP"])
            df["AP_delta"] = df["AP"] - clean_ap
            df["AP_delta_%"] = df["AP_delta"] / clean_ap * 100 if clean_ap else 0

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("AP без искажений", format_float(clean_ap))
            with c2:
                worst = df.sort_values("AP_delta").iloc[0]
                metric_card("Худший режим", worst.get("mode"))
            with c3:
                metric_card("Падение AP", format_float(worst.get("AP_delta")))

            if px:
                fig = px.bar(df.sort_values("AP"), x="mode", y="AP", title="AP при разных искажениях", text="AP")
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                fig = px.bar(df.sort_values("AP_delta"), x="mode", y="AP_delta", title="Изменение AP относительно режима без искажений")
                st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Этот блок показывает, что модель проверялась не только на чистых данных. "
        "Наиболее опасные искажения: blur и noise."
    )


def page_segmentation(root: Path) -> None:
    st.header("YOLO-Seg: переход от рамок к маскам")
    df = load_table(root, "D2S_YOLO_SEG_last")
    if df is None:
        st.warning("Файл D2S_YOLO_SEG_last.csv не найден.")
        return

    show_dataframe(df)
    if not df.empty:
        row = df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Точность bbox", format_float(row.get("P_box")))
        with c2:
            metric_card("Полнота bbox", format_float(row.get("R_box")))
        with c3:
            metric_card("mAP50-95 bbox", format_float(row.get("mAP5095_box")))
        with c4:
            metric_card("mAP50-95 mask", format_float(row.get("mAP5095_mask")))

    if px:
        metrics = [c for c in ["mAP50_box", "mAP5095_box", "mAP50_mask", "mAP5095_mask"] if c in df.columns]
        if metrics and not df.empty:
            vals = pd.DataFrame({"Метрика": metrics, "Значение": [float(df.iloc[0][m]) for m in metrics]})
            fig = px.bar(vals, x="Метрика", y="Значение", title="Качество рамок и масок YOLO-Seg", text="Значение")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Этот блок связывает проект с ВКР: после детекции можно перейти к сегментации товаров."
    )


def page_gallery(root: Path) -> None:
    st.header("Визуальные примеры")
    images = find_images(root, max_images=300)
    if not images:
        st.warning(
            "Готовые изображения не найдены. Можно сформировать их скриптом "
            "scripts/final_showcase_select_best.py или пересобрать отчётные графики."
        )
        return

    st.write(f"Найдено изображений: {len(images)}")
    folders = sorted({str(p.parent.relative_to(root)) if root in p.parents else str(p.parent) for p in images})
    selected_folder = st.selectbox("Папка", ["Все"] + folders)
    if selected_folder != "Все":
        images = [p for p in images if str(p.parent.relative_to(root)) == selected_folder]

    limit = st.slider("Сколько изображений показать", 1, min(60, len(images)), min(12, len(images)))
    cols = st.columns(3)
    for idx, img_path in enumerate(images[:limit]):
        with cols[idx % 3]:
            st.image(str(img_path), caption=str(img_path.relative_to(root)) if root in img_path.parents else str(img_path), use_container_width=True)

    st.info(
        "На этой вкладке удобно демонстрировать преподавателю не только таблицы, "
        "но и реальные картинки с графиками или предсказаниями моделей."
    )


def page_hardware(root: Path) -> None:
    st.header("Аппаратный и информационно-программный комплекс")

    rows = [
        {"Компонент": "Операционная система", "Значение": platform.platform()},
        {"Компонент": "Python", "Значение": sys.version.split()[0]},
        {"Компонент": "Процессор", "Значение": platform.processor() or "не определён"},
    ]
    if psutil:
        rows.append({"Компонент": "ОЗУ", "Значение": f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)} ГБ"})
        rows.append({"Компонент": "Логических CPU", "Значение": psutil.cpu_count(logical=True)})
    else:
        rows.append({"Компонент": "psutil", "Значение": "не установлен"})

    if torch:
        cuda = torch.cuda.is_available()
        rows.append({"Компонент": "PyTorch", "Значение": torch.__version__})
        rows.append({"Компонент": "CUDA", "Значение": "доступна" if cuda else "не доступна"})
        if cuda:
            rows.append({"Компонент": "GPU", "Значение": torch.cuda.get_device_name(0)})
    else:
        rows.append({"Компонент": "PyTorch", "Значение": "не установлен"})

    show_dataframe(pd.DataFrame(rows), height=260)

    st.subheader("Программные компоненты")
    show_dataframe(package_status(), height=420)

    st.subheader("Что демонстрируется на зачёте/защите")
    st.markdown(
        """
        - проект ShelfVision как программная основа;
        - CSV-таблицы с результатами экспериментов;
        - графики качества, скорости и устойчивости;
        - визуальные примеры предсказаний;
        - аппаратная среда, Python и библиотеки обработки данных;
        - воспроизводимые команды запуска.
        """
    )


def page_commands(root: Path) -> None:
    st.header("Команды и воспроизводимость")

    st.subheader("Запуск интерфейса")
    st.code("streamlit run scripts/interface_app.py", language="bash")

    st.subheader("Установка зависимостей")
    st.code("pip install -r requirements.txt", language="bash")

    st.subheader("Пересборка отчётных таблиц и графиков")
    st.code(
        "python scripts/make_master_report.py --project_root . --out_dir reports/course_project",
        language="bash",
    )

    st.subheader("Визуальные примеры моделей")
    st.code(
        "python scripts/final_showcase_select_best.py --paths_json reports/path_scan/paths.json --out_dir artifacts/final_showcase --imgsz 640 --conf 0.25 --limit 20 --device 0",
        language="bash",
    )

    make_report = root / "scripts" / "make_master_report.py"
    if make_report.exists():
        if st.button("Запустить make_master_report.py"):
            with st.spinner("Сборка отчётных материалов..."):
                code, out = run_command(
                    [sys.executable, str(make_report), "--project_root", ".", "--out_dir", "reports/course_project"],
                    cwd=root,
                )
            st.code(out, language="text")
            if code == 0:
                st.success("Отчётные материалы пересобраны.")
            else:
                st.error("Команда завершилась с ошибкой.")
    else:
        st.warning("scripts/make_master_report.py не найден.")


# -----------------------------------------------------------------------------
# Основная функция
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")
    root = repo_root()
    render_sidebar(root)

    st.title(APP_TITLE)
    st.caption("Интерфейс добавлен в основной проект для демонстрации экспериментов и задела для ВКР.")

    tabs = st.tabs([
        "Главная",
        "Детекция",
        "YOLO-абляции",
        "WBF",
        "Устойчивость",
        "YOLO-Seg",
        "Визуальные примеры",
        "АПК",
        "Команды",
    ])

    with tabs[0]:
        page_home(root)
    with tabs[1]:
        page_detection(root)
    with tabs[2]:
        page_yolo_ablations(root)
    with tabs[3]:
        page_wbf(root)
    with tabs[4]:
        page_robustness(root)
    with tabs[5]:
        page_segmentation(root)
    with tabs[6]:
        page_gallery(root)
    with tabs[7]:
        page_hardware(root)
    with tabs[8]:
        page_commands(root)


if __name__ == "__main__":
    main()
