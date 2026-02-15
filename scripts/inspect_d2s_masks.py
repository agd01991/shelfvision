# scripts/inspect_d2s_masks.py
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_dir", type=str, required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    masks_dir = Path(args.masks_dir)
    paths = sorted([p for p in masks_dir.rglob("*") if p.suffix.lower() in (".png", ".tif", ".tiff")])

    if not paths:
        raise SystemExit(f"No mask files found in: {masks_dir}")

    mins, maxs = [], []
    sample_vals = Counter()

    for p in paths[: args.limit]:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        vals = np.unique(m)
        mins.append(int(vals.min()))
        maxs.append(int(vals.max()))
        # берём “не нули”
        for v in vals[:200]:
            if int(v) != 0:
                sample_vals[int(v)] += 1

        print(f"{p.name}: dtype={m.dtype} shape={m.shape} min={int(vals.min())} max={int(vals.max())} uniq={len(vals)}")

    gmin = min(mins) if mins else None
    gmax = max(maxs) if maxs else None

    print("\n=== SUMMARY ===")
    print(f"files_checked={min(args.limit, len(paths))} global_min={gmin} global_max={gmax}")
    top = sample_vals.most_common(30)
    print("top_nonzero_values:", top)

    # эвристика
    if gmax is not None and gmax >= 1000:
        print("\nHeuristic: looks like values >= 1000 exist -> possible encoding class*1000+instance")
    elif gmax is not None and gmax <= 255:
        print("\nHeuristic: looks like <=255 -> could be palette/class-id, or separate instance map")
    else:
        print("\nHeuristic: unclear encoding")


if __name__ == "__main__":
    main()
