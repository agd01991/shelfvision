from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco_path = Path(args.coco_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    ids = [int(im["id"]) for im in images]
    random.seed(args.seed)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_test = n - n_train - n_val

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])

    img_by_id = {int(im["id"]): im for im in images}
    anns_by_img = {}
    for a in anns:
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)

    def dump(split_name: str, split_ids: set[int]):
        split_images = [img_by_id[i] for i in split_ids if i in img_by_id]
        split_anns = []
        for i in split_ids:
            split_anns.extend(anns_by_img.get(i, []))
        out = {"images": split_images, "annotations": split_anns, "categories": cats}
        (out_dir / f"{split_name}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(split_name, "images=", len(split_images), "anns=", len(split_anns))

    dump("train", train_ids)
    dump("val", val_ids)
    dump("test", test_ids)

    (out_dir / "splits.json").write_text(
        json.dumps({"train": list(train_ids), "val": list(val_ids), "test": list(test_ids), "seed": args.seed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()