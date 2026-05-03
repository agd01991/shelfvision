#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Демонстрационный скрипт для зачета.
Запускать из корня проекта ShelfVision:
    python scripts/demo_exam_showcase.py --open
или если файл лежит в корне:
    python demo_exam_showcase.py --open --project-root .

Скрипт не запускает тяжелое обучение заново. Он показывает, что эксперименты есть:
читает CSV-таблицы, строит графики, ищет готовые картинки и открывает HTML-отчет.
"""

from __future__ import annotations

import argparse
import html
import platform
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except Exception as exc:
    raise SystemExit("Не установлен pandas. Выполните: pip install pandas matplotlib\n" + str(exc))

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("Не установлен matplotlib. Выполните: pip install pandas matplotlib\n" + str(exc))

CSV_FILES = {
    "DIR1_models": "reports/all_stats/DIR1_models.csv",
    "YOLO_11_ablations": "reports/all_stats/YOLO_11_ablations.csv",
    "DIR3_WBF": "reports/all_stats/DIR3_WBF.csv",
    "DIR5_robustness": "reports/all_stats/DIR5_robustness.csv",
    "D2S_YOLO_SEG_last": "reports/all_stats/D2S_YOLO_SEG_last.csv",
    "OVERALL_detection_min": "reports/all_stats/OVERALL_detection_min.csv",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "scripts":
        return here.parent.parent
    return Path.cwd().resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Не удалось прочитать {path}: {exc}")
        return None


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def get_gpu_info() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "; ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        return "CUDA не обнаружена через torch"
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().replace("\n", "; ")
    except Exception:
        pass

    return "GPU не определена или не требуется для просмотра готовых результатов"


def get_ram_info() -> str:
    try:
        import psutil  # type: ignore
        return f"{psutil.virtual_memory().total / (1024 ** 3):.1f} ГБ"
    except Exception:
        return "не определено"


def environment_info(root: Path) -> dict[str, str]:
    return {
        "Дата запуска": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "Корень проекта": str(root),
        "Python": sys.version.split()[0],
        "ОС": f"{platform.system()} {platform.release()}",
        "Процессор": platform.processor() or "не определено",
        "ОЗУ": get_ram_info(),
        "GPU": get_gpu_info(),
    }


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["model", "system", "exp", "mode", "name"]:
        if col in df.columns:
            return col
    return None


def save_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path, rotate: int = 25) -> Optional[Path]:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    d = df[[x_col, y_col]].copy()
    d[y_col] = to_num(d[y_col])
    d = d.dropna(subset=[y_col])
    if d.empty:
        return None
    plt.figure(figsize=(10, 5))
    plt.bar(d[x_col].astype(str), d[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, label_col: str, title: str, out_path: Path) -> Optional[Path]:
    if not {x_col, y_col, label_col}.issubset(df.columns):
        return None
    d = df[[x_col, y_col, label_col]].copy()
    d[x_col] = to_num(d[x_col])
    d[y_col] = to_num(d[y_col])
    d = d.dropna(subset=[x_col, y_col])
    if d.empty:
        return None
    plt.figure(figsize=(8, 5))
    plt.scatter(d[x_col], d[y_col])
    for _, row in d.iterrows():
        plt.annotate(str(row[label_col]), (row[x_col], row[y_col]), fontsize=8)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def build_plots(data: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    plots_dir = ensure_dir(out_dir / "plots")
    plots: list[Path] = []

    df = data.get("DIR1_models")
    if df is not None:
        label = find_label_column(df)
        if label and "AP50-95" in df.columns:
            p = save_bar(df, label, "AP50-95", "Сравнение моделей детекции по AP50-95", plots_dir / "01_models_ap5095.png")
            if p:
                plots.append(p)
        if label and {"AP50-95", "ms_per_image"}.issubset(df.columns):
            p = save_scatter(df, "ms_per_image", "AP50-95", label, "Качество и скорость моделей", plots_dir / "02_quality_speed.png")
            if p:
                plots.append(p)

    df = data.get("YOLO_11_ablations")
    if df is not None and {"exp", "mAP50-95"}.issubset(df.columns):
        p = save_bar(df, "exp", "mAP50-95", "Абляционный анализ YOLO", plots_dir / "03_yolo_ablations.png", rotate=45)
        if p:
            plots.append(p)

    df = data.get("DIR3_WBF")
    if df is not None and {"system", "AP"}.issubset(df.columns):
        p = save_bar(df, "system", "AP", "YOLO, RT-DETR и WBF по AP", plots_dir / "04_wbf_ap.png")
        if p:
            plots.append(p)

    df = data.get("DIR5_robustness")
    if df is not None and {"mode", "AP"}.issubset(df.columns):
        p = save_bar(df, "mode", "AP", "Устойчивость к искажениям", plots_dir / "05_robustness_ap.png")
        if p:
            plots.append(p)

    df = data.get("OVERALL_detection_min")
    if df is not None and {"system", "AP50-95"}.issubset(df.columns):
        p = save_bar(df, "system", "AP50-95", "Итоговая сводка AP50-95", plots_dir / "06_overall_ap5095.png")
        if p:
            plots.append(p)

    return plots


def find_images(root: Path, limit: int) -> list[Path]:
    folders = [
        root / "artifacts/final_showcase/best_demo",
        root / "artifacts/final_showcase/all_models",
        root / "reports/course_project/plots",
        root / "reports/master/plots",
        root / "artifacts",
        root / "reports",
    ]
    images: list[Path] = []
    seen: set[Path] = set()
    for folder in folders:
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p not in seen:
                images.append(p)
                seen.add(p)
                if len(images) >= limit:
                    return images
    return images


def rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except Exception:
        return path.as_posix()


def df_html(df: pd.DataFrame, max_rows: int = 8) -> str:
    d = df.head(max_rows).copy()
    for col in d.columns:
        if pd.api.types.is_numeric_dtype(d[col]):
            d[col] = d[col].map(lambda x: f"{x:.5f}" if pd.notna(x) else "")
    return d.to_html(index=False, border=0, escape=True)


def badge(ok: bool) -> str:
    return '<span class="ok">НАЙДЕНО</span>' if ok else '<span class="bad">НЕТ ФАЙЛА</span>'


def summary_html(data: dict[str, pd.DataFrame]) -> str:
    items: list[str] = []

    overall = data.get("OVERALL_detection_min")
    if overall is not None and {"system", "AP50-95"}.issubset(overall.columns):
        d = overall.copy()
        d["AP50-95"] = to_num(d["AP50-95"])
        best = d.dropna(subset=["AP50-95"]).sort_values("AP50-95", ascending=False).head(1)
        if not best.empty:
            r = best.iloc[0]
            items.append(f"<li><b>Лучший итоговый подход:</b> {html.escape(str(r.get('system', '')))} {html.escape(str(r.get('name', '')))}, AP50-95 = {r['AP50-95']:.5f}.</li>")

    yolo = data.get("YOLO_11_ablations")
    if yolo is not None and {"exp", "mAP50-95"}.issubset(yolo.columns):
        d = yolo.copy()
        d["mAP50-95"] = to_num(d["mAP50-95"])
        best = d.dropna(subset=["mAP50-95"]).sort_values("mAP50-95", ascending=False).head(1)
        if not best.empty:
            r = best.iloc[0]
            items.append(f"<li><b>Лучшая YOLO-конфигурация:</b> {html.escape(str(r['exp']))}, mAP50-95 = {r['mAP50-95']:.5f}.</li>")

    rob = data.get("DIR5_robustness")
    if rob is not None and {"mode", "AP"}.issubset(rob.columns):
        d = rob.copy()
        d["AP"] = to_num(d["AP"])
        clean = d[d["mode"].astype(str).str.lower() == "clean"]
        if not clean.empty:
            clean_ap = float(clean.iloc[0]["AP"])
            d["delta"] = d["AP"] - clean_ap
            worst = d.sort_values("delta").head(1)
            if not worst.empty:
                r = worst.iloc[0]
                items.append(f"<li><b>Самое критичное искажение:</b> {html.escape(str(r['mode']))}, изменение AP = {r['delta']:.5f}.</li>")

    seg = data.get("D2S_YOLO_SEG_last")
    if seg is not None and "mAP5095_mask" in seg.columns:
        vals = pd.to_numeric(seg["mAP5095_mask"], errors="coerce").dropna()
        if not vals.empty:
            items.append(f"<li><b>YOLO-Seg:</b> mAP50-95 mask = {vals.iloc[0]:.5f}.</li>")

    if not items:
        return "<p>Сводные выводы не сформированы: не найдены нужные таблицы.</p>"
    return "<ul>" + "\n".join(items) + "</ul>"


def build_html(root: Path, out_dir: Path, data: dict[str, pd.DataFrame], statuses: dict[str, bool], plots: list[Path], images: list[Path]) -> Path:
    rows_env = "\n".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in environment_info(root).items())
    rows_status = "\n".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(CSV_FILES[k])}</td><td>{badge(v)}</td></tr>" for k, v in statuses.items())

    plot_html = "".join(f'<figure><img src="{html.escape(rel(p, out_dir))}"><figcaption>{html.escape(p.name)}</figcaption></figure>' for p in plots)
    img_html = "".join(f'<figure><img src="{html.escape(rel(p, out_dir))}"><figcaption>{html.escape(rel(p, root))}</figcaption></figure>' for p in images)

    tables = ""
    for name, df in data.items():
        tables += f"<section class='card'><h3>{html.escape(name)}</h3><p>{df.shape[0]} строк, {df.shape[1]} столбцов</p>{df_html(df)}</section>"

    text = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Демонстрация экспериментов ShelfVision</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; color: #222; }}
h1,h2,h3 {{ color: #1f2937; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
table {{ border-collapse: collapse; width: 100%; background: white; margin: 8px 0; }}
th,td {{ border: 1px solid #ddd; padding: 7px; font-size: 13px; vertical-align: top; }}
th {{ background: #eee; }}
.ok {{ color: #0a7a2f; font-weight: bold; }}
.bad {{ color: #b00020; font-weight: bold; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 16px; }}
figure {{ background: white; padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin: 0; }}
figure img {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-size: 12px; margin-top: 6px; color: #555; word-break: break-word; }}
code {{ background: #eee; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Демонстрация экспериментов ShelfVision</h1>
<section class="card"><h2>Аппаратный и программный комплекс</h2><table>{rows_env}</table></section>
<section class="card"><h2>Проверка файлов экспериментов</h2><table><tr><th>Блок</th><th>Файл</th><th>Статус</th></tr>{rows_status}</table></section>
<section class="card"><h2>Ключевые выводы</h2>{summary_html(data)}</section>
<section class="card"><h2>Графики</h2><div class="grid">{plot_html or '<p>Графики не построены.</p>'}</div></section>
<section class="card"><h2>Визуальные примеры</h2><p>Если заранее создана папка <code>artifacts/final_showcase</code>, здесь будут показаны примеры предсказаний моделей.</p><div class="grid">{img_html or '<p>Картинки не найдены. Для демонстрации можно заранее запустить scripts/final_showcase_select_best.py.</p>'}</div></section>
<section class="card"><h2>Предпросмотр таблиц</h2>{tables}</section>
</body>
</html>"""
    path = out_dir / "index.html"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(project_root_from_script()), help="Корень проекта ShelfVision")
    parser.add_argument("--out-dir", default="demo_showcase", help="Папка для HTML-демонстрации")
    parser.add_argument("--open", action="store_true", help="Открыть страницу в браузере")
    parser.add_argument("--image-limit", type=int, default=24, help="Сколько картинок показать")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = ensure_dir((root / args.out_dir).resolve())

    print("=" * 80)
    print("Демонстрация экспериментов ShelfVision")
    print("=" * 80)
    print("Корень проекта:", root)
    print("Папка отчета:", out_dir)
    print()

    data: dict[str, pd.DataFrame] = {}
    statuses: dict[str, bool] = {}

    for name, rel_path in CSV_FILES.items():
        path = root / rel_path
        statuses[name] = path.exists()
        print(f"[{'OK' if path.exists() else 'NO'}] {name}: {path}")
        df = read_csv_safe(path)
        if df is not None:
            data[name] = df

    print("\nСтрою графики...")
    plots = build_plots(data, out_dir)
    for p in plots:
        print("[PLOT]", p)

    print("\nИщу визуальные примеры...")
    images = find_images(root, args.image_limit)
    for p in images[:10]:
        print("[IMG]", p)
    if len(images) > 10:
        print(f"... и еще {len(images) - 10} изображений")

    html_path = build_html(root, out_dir, data, statuses, plots, images)
    print("\nГотово:", html_path)

    if args.open:
        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
