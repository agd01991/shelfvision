from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    prepared = Path(args.prepared_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads((prepared / "annotations.json").read_text(encoding="utf-8"))
    splits = json.loads((prepared / "splits.json").read_text(encoding="utf-8"))

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    img_by_id = {im["id"]: im for im in images}
    anns_by_img = {}
    for a in anns:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    for split_name in ("train", "val", "test"):
        ids = set(int(x) for x in splits.get(split_name, []))
        split_images = [img_by_id[i] for i in ids if i in img_by_id]
        split_anns = []
        for i in ids:
            split_anns.extend(anns_by_img.get(i, []))

        out = {"images": split_images, "annotations": split_anns, "categories": cats}
        (out_dir / f"{split_name}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8"
        )
        print(split_name, "images=", len(split_images), "anns=", len(split_anns))


if __name__ == "__main__":
    main()
