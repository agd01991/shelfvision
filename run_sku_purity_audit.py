from __future__ import annotations

import argparse

from src.identification.sku_purity_audit import run_sku_purity_audit


OUTPUT_LABELS_RU = {
    "summary_json": "JSON-сводка аудита",
    "ref_purity_csv": "CSV чистоты эталонов",
    "mixed_sku_candidates_csv": "CSV кандидатов на смешанные SKU",
    "ref_outlier_candidates_csv": "CSV выбивающихся эталонов",
    "report_md": "Markdown-отчёт аудита",
    "contact_sheets_dir": "Папка контактных листов",
}


def _label_output(name: str) -> str:
    return OUTPUT_LABELS_RU.get(str(name), str(name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Аудит чистоты существующей SKU-галереи")
    parser.add_argument("--gallery-dir", required=True, help="Папка SKU-галереи")
    parser.add_argument("--out-dir", required=True, help="Папка результатов аудита")
    parser.add_argument("--own-centroid-threshold", type=float, default=0.65, help="Порог сходства эталона с центром своего SKU")
    parser.add_argument("--own-mean-threshold", type=float, default=0.60, help="Порог среднего сходства эталона внутри своего SKU")
    parser.add_argument("--other-margin", type=float, default=0.08, help="Минимальный отрыв от ближайшего другого SKU")
    parser.add_argument("--min-other-similarity", type=float, default=0.68, help="Минимальное сходство с другим SKU для подозрительного эталона")
    parser.add_argument("--max-refs-per-sku", type=int, default=50, help="Максимум эталонов на один SKU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs = run_sku_purity_audit(
        gallery_dir=args.gallery_dir,
        out_dir=args.out_dir,
        own_centroid_threshold=args.own_centroid_threshold,
        own_mean_threshold=args.own_mean_threshold,
        other_margin=args.other_margin,
        min_other_similarity=args.min_other_similarity,
        max_refs_per_sku=args.max_refs_per_sku,
    )

    print("=== ShelfVision: аудит чистоты SKU завершён ===", flush=True)
    for name, path in outputs.items():
        print(f"{_label_output(name)}: {path}", flush=True)


if __name__ == "__main__":
    main()
