from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_TITLE = "ShelfVision: итоговый мини-отчёт"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _read_json(path: str | Path | None) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _read_csv(path: str | Path | None) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def _collect_images(folder: str | Path | None, limit: int = 8) -> List[Path]:
    if not folder:
        return []
    p = Path(folder)
    if not p.exists():
        return []
    images = sorted(item for item in p.rglob("*") if item.suffix.lower() in IMAGE_EXTS and item.is_file())
    return images[:limit]


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _df_to_markdown(df: Optional[pd.DataFrame], max_rows: int = 8) -> str:
    if df is None or df.empty:
        return "Нет данных."
    return df.head(max_rows).to_markdown(index=False)


def _df_to_html(df: Optional[pd.DataFrame], max_rows: int = 8) -> str:
    if df is None or df.empty:
        return "<p>Нет данных.</p>"
    return df.head(max_rows).to_html(index=False, classes="table", border=0)


def _build_markdown(
    title: str,
    comparison_json: Optional[Dict[str, Any]],
    comparison_csv: Optional[pd.DataFrame],
    recommendation_json: Optional[Dict[str, Any]],
    density_json: Optional[Dict[str, Any]],
    density_csv: Optional[pd.DataFrame],
    images: List[Path],
    out_dir: Path,
) -> str:
    lines: List[str] = [f"# {title}", ""]

    lines.extend([
        "## 1. Назначение системы",
        "",
        "ShelfVision предназначена для анализа изображений товарных полок: система обнаруживает товары, визуализирует результаты, считает метрики качества и помогает выбрать наиболее подходящий pipeline.",
        "",
    ])

    best_model = None
    best_score = None
    highlights: List[str] = []
    if comparison_json and "comparison" in comparison_json:
        comp = comparison_json["comparison"]
        best_model = comp.get("best_model")
        best_score = comp.get("best_score")
        highlights = comp.get("highlights", []) or []
    elif recommendation_json and "best_model" in recommendation_json:
        best = recommendation_json["best_model"]
        best_model = best.get("model_name")
        best_score = best.get("score")
        reason = best.get("reason")
        if reason:
            highlights = [reason]

    lines.extend(["## 2. Рекомендуемый pipeline", ""])
    if best_model:
        lines.append(f"**Рекомендуемая модель:** {best_model}")
        if best_score is not None:
            lines.append(f"**Интегральный score:** {float(best_score):.4f}")
        lines.append("")
    else:
        lines.append("Данные рекомендации не найдены.")
        lines.append("")

    if highlights:
        lines.append("Ключевые выводы:")
        for item in highlights:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend([
        "## 3. Сравнение моделей",
        "",
        _df_to_markdown(comparison_csv, max_rows=10),
        "",
    ])

    lines.extend(["## 4. Анализ плотности товаров", ""])
    if density_json:
        total_objects = density_json.get("total_objects", 0)
        images_count = density_json.get("images_count", 0)
        densest_zone = density_json.get("densest_zone")
        lines.append(f"Обработано изображений: **{images_count}**.")
        lines.append(f"Найдено объектов: **{total_objects}**.")
        if densest_zone:
            lines.append(
                f"Наиболее плотная зона: **{densest_zone.get('zone_name', '-')}**, объектов: **{densest_zone.get('objects_count', '-')}**."
            )
        lines.append("")
    lines.append(_df_to_markdown(density_csv, max_rows=9))
    lines.append("")

    lines.extend(["## 5. Визуальные примеры", ""])
    if images:
        for image_path in images:
            rel_path = _rel(image_path, out_dir)
            lines.append(f"![{image_path.name}]({rel_path})")
            lines.append("")
    else:
        lines.append("Изображения для демонстрации не найдены.")
        lines.append("")

    lines.extend([
        "## 6. Что можно показать на защите",
        "",
        "- загрузку изображения и запуск модели через интерфейс;",
        "- визуализацию найденных товаров;",
        "- таблицу найденных объектов;",
        "- сравнение YOLO, RT-DETR-L, Faster R-CNN и WBF;",
        "- рекомендацию лучшего pipeline;",
        "- анализ плотности товаров по зонам полки.",
        "",
    ])

    return "\n".join(lines)


def _build_html(
    title: str,
    comparison_json: Optional[Dict[str, Any]],
    comparison_csv: Optional[pd.DataFrame],
    recommendation_json: Optional[Dict[str, Any]],
    density_json: Optional[Dict[str, Any]],
    density_csv: Optional[pd.DataFrame],
    images: List[Path],
    out_dir: Path,
) -> str:
    best_model = "Нет данных"
    best_score = "-"
    highlights: List[str] = []

    if comparison_json and "comparison" in comparison_json:
        comp = comparison_json["comparison"]
        best_model = str(comp.get("best_model", "Нет данных"))
        if comp.get("best_score") is not None:
            best_score = f"{float(comp.get('best_score')):.4f}"
        highlights = comp.get("highlights", []) or []
    elif recommendation_json and "best_model" in recommendation_json:
        best = recommendation_json["best_model"]
        best_model = str(best.get("model_name", "Нет данных"))
        if best.get("score") is not None:
            best_score = f"{float(best.get('score')):.4f}"
        if best.get("reason"):
            highlights = [best["reason"]]

    density_summary = "Нет данных"
    if density_json:
        density_summary = (
            f"Обработано изображений: {density_json.get('images_count', 0)}. "
            f"Найдено объектов: {density_json.get('total_objects', 0)}."
        )
        densest = density_json.get("densest_zone")
        if densest:
            density_summary += f" Самая плотная зона: {densest.get('zone_name', '-')} ({densest.get('objects_count', '-')})."

    image_html = ""
    for image_path in images:
        rel_path = _rel(image_path, out_dir)
        image_html += f'<figure><img src="{rel_path}" alt="{image_path.name}"><figcaption>{image_path.name}</figcaption></figure>'
    if not image_html:
        image_html = "<p>Изображения для демонстрации не найдены.</p>"

    highlights_html = "".join(f"<li>{item}</li>" for item in highlights) or "<li>Ключевые выводы не найдены.</li>"

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; background: #f9fafb; }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    .table th, .table td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    .gallery {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; border-radius: 12px; border: 1px solid #e5e7eb; }}
    figcaption {{ color: #6b7280; font-size: 13px; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>ShelfVision предназначена для анализа изображений товарных полок: система обнаруживает товары, визуализирует результаты, считает метрики качества и помогает выбрать наиболее подходящий pipeline.</p>

  <div class="cards">
    <div class="card"><div>Рекомендуемый pipeline</div><div class="metric">{best_model}</div></div>
    <div class="card"><div>Интегральный score</div><div class="metric">{best_score}</div></div>
    <div class="card"><div>Плотность</div><div>{density_summary}</div></div>
  </div>

  <h2>Ключевые выводы</h2>
  <ul>{highlights_html}</ul>

  <h2>Сравнение моделей</h2>
  {_df_to_html(comparison_csv, max_rows=10)}

  <h2>Анализ плотности товаров</h2>
  {_df_to_html(density_csv, max_rows=9)}

  <h2>Визуальные примеры</h2>
  <div class="gallery">{image_html}</div>

  <h2>Что можно показать на защите</h2>
  <ul>
    <li>загрузку изображения и запуск модели через интерфейс;</li>
    <li>визуализацию найденных товаров;</li>
    <li>таблицу найденных объектов;</li>
    <li>сравнение YOLO, RT-DETR-L, Faster R-CNN и WBF;</li>
    <li>рекомендацию лучшего pipeline;</li>
    <li>анализ плотности товаров по зонам полки.</li>
  </ul>
</body>
</html>"""


def build_mini_report(
    out_dir: str | Path = "results/mini_report",
    title: str = DEFAULT_TITLE,
    comparison_json: str | Path | None = None,
    comparison_csv: str | Path | None = None,
    recommendation_json: str | Path | None = None,
    density_json: str | Path | None = None,
    density_csv: str | Path | None = None,
    images_dir: str | Path | None = None,
    image_limit: int = 8,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_data = _read_json(comparison_json)
    recommendation_data = _read_json(recommendation_json)
    density_data = _read_json(density_json)
    comparison_table = _read_csv(comparison_csv)
    density_table = _read_csv(density_csv)
    images = _collect_images(images_dir, limit=image_limit)

    md = _build_markdown(
        title=title,
        comparison_json=comparison_data,
        comparison_csv=comparison_table,
        recommendation_json=recommendation_data,
        density_json=density_data,
        density_csv=density_table,
        images=images,
        out_dir=out_dir,
    )
    html = _build_html(
        title=title,
        comparison_json=comparison_data,
        comparison_csv=comparison_table,
        recommendation_json=recommendation_data,
        density_json=density_data,
        density_csv=density_table,
        images=images,
        out_dir=out_dir,
    )

    md_path = out_dir / "mini_report.md"
    html_path = out_dir / "mini_report.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    manifest = {
        "title": title,
        "comparison_json": str(comparison_json) if comparison_json else None,
        "comparison_csv": str(comparison_csv) if comparison_csv else None,
        "recommendation_json": str(recommendation_json) if recommendation_json else None,
        "density_json": str(density_json) if density_json else None,
        "density_csv": str(density_csv) if density_csv else None,
        "images_dir": str(images_dir) if images_dir else None,
        "images_used": [str(path) for path in images],
        "outputs": {"markdown": str(md_path), "html": str(html_path)},
    }
    manifest_path = out_dir / "mini_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"markdown": md_path, "html": html_path, "manifest": manifest_path}
