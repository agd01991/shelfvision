from __future__ import annotations

import argparse

from src.identification.sku_similarity_audit import run_sku_similarity_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Аудит визуально похожих SKU в существующей галерее")
    parser.add_argument("--gallery-dir", required=True, help="Папка SKU-галереи")
    parser.add_argument("--out-dir", required=True, help="Папка результатов аудита")
    parser.add_argument("--pair-report-threshold", type=float, default=0.75, help="Порог попадания пары SKU в отчёт")
    parser.add_argument("--candidate-threshold", type=float, default=0.82, help="Порог кандидата на ручную проверку")
    parser.add_argument("--top-n", type=int, default=100, help="Сколько лучших пар сохранить")
    parser.add_argument("--contact-sheet-limit", type=int, default=50, help="Сколько контактных листов сформировать")
    parser.add_argument("--max-refs-per-sku", type=int, default=10, help="Максимум эталонов на один SKU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_sku_similarity_audit(
        gallery_dir=args.gallery_dir,
        out_dir=args.out_dir,
        pair_report_threshold=args.pair_report_threshold,
        candidate_threshold=args.candidate_threshold,
        top_n=args.top_n,
        contact_sheet_limit=args.contact_sheet_limit,
        max_refs_per_sku=args.max_refs_per_sku,
    )

    print("=== ShelfVision: аудит похожих SKU завершён ===", flush=True)
    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
