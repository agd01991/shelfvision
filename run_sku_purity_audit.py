from __future__ import annotations

import argparse

from src.identification.sku_purity_audit import run_sku_purity_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SKU purity audit for an existing SKU gallery")
    parser.add_argument("--gallery-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--own-centroid-threshold", type=float, default=0.65)
    parser.add_argument("--own-mean-threshold", type=float, default=0.60)
    parser.add_argument("--other-margin", type=float, default=0.08)
    parser.add_argument("--min-other-similarity", type=float, default=0.68)
    parser.add_argument("--max-refs-per-sku", type=int, default=50)
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

    print("=== SKU purity audit done ===", flush=True)
    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
