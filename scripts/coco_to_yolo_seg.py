from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_coco(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def find_image(images_root: Path, file_name: str) -> Path | None:
    # file_name в D2S_small у нас относительный, но может быть с подпапками
    p = images_root / file_name
    if p.exists():
        return p
    p2 = images_root / Path(file_name).name
    if p2.exists():
        return p2
    return None


def poly_to_yolo_line(poly: List[float], w: int, h: int, cls: int) -> str | None:
    # poly = [x1,y1,x2,y2,...]  (пиксели)
    if not poly or len(poly) < 6 or len(poly) % 2 != 0:
        return None

    pts = []
    for i in range(0, len(poly), 2):
        x = float(poly[i]) / float(w)
        y = float(poly[i + 1]) / float(h)
        # clamp
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        pts.append(f"{x:.6f} {y:.6f}")

    return f"{cls} " + " ".join(pts)


def convert_split(
    coco_json: Path,
    images_root: Path,
    out_images: Path,
    out_labels: Path,
    catid_to_idx: Dict[int, int],
) -> Tuple[int, int]:
    coco = load_coco(coco_json)
    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    img_by_id = {int(im["id"]): im for im in images}
    anns_by_img: Dict[int, List[dict]] = {}
    for a in anns:
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)

    kept_imgs = 0
    kept_anns = 0

    for im in images:
        img_id = int(im["id"])
        fn = im["file_name"]
        w = int(im.get("width", 0))
        h = int(im.get("height", 0))

        src = find_image(images_root, fn)
        if src is None:
            continue

        dst_img = out_images / Path(fn).name
        link_or_copy(src, dst_img)
        kept_imgs += 1

        # labels
        lines: List[str] = []
        for a in anns_by_img.get(img_id, []):
            cat_id = int(a["category_id"])
            cls = catid_to_idx.get(cat_id)
            if cls is None:
                continue

            # --- segmentation normalization ---
            seg = a.get("segmentation", None)
            polys = []

            # 1) COCO polygon as flat list: [x1,y1,x2,y2,...]
            if isinstance(seg, list) and seg and isinstance(seg[0], (int, float)):
                polys = [seg]

            # 2) COCO polygon as list of polygons: [[...], [...]]
            elif isinstance(seg, list) and seg and isinstance(seg[0], list):
                polys = seg

            # 3) RLE dict -> convert to polygons (requires pycocotools + cv2)
            elif isinstance(seg, dict):
                try:
                    from pycocotools import mask as mask_utils
                    import numpy as np
                    import cv2

                    m = mask_utils.decode(seg)  # HxW or HxWxN
                    if m.ndim == 3:
                        m = m[:, :, 0]
                    m = (m > 0).astype("uint8") * 255

                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cnt.shape[0] < 3:
                            continue
                        cnt = cnt.reshape(-1, 2).astype(float).tolist()
                        flat = []
                        for x, y in cnt:
                            flat.extend([x, y])
                        if len(flat) >= 6:
                            polys.append(flat)
                except Exception:
                    polys = []

            # если полигонов нет — пропускаем
            if not polys:
                continue

            for poly in polys:
                if not isinstance(poly, list):
                    continue
                line = poly_to_yolo_line(poly, w, h, cls)
                if line:
                    lines.append(line)
                    kept_anns += 1

        dst_lbl = (out_labels / dst_img.stem).with_suffix(".txt")
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return kept_imgs, kept_anns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True, help="data/raw/d2s_small/images")
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--out_dir", required=True, help="data/yolo_cache/d2s_small_seg")
    args = ap.parse_args()

    images_root = Path(args.images_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    for s in ["train", "val", "test"]:
        (out_images / s).mkdir(parents=True, exist_ok=True)
        (out_labels / s).mkdir(parents=True, exist_ok=True)

    # категории берём из train_json
    coco_train = load_coco(Path(args.train_json))
    cats = coco_train.get("categories", [])
    cats_sorted = sorted(cats, key=lambda c: int(c["id"]))
    catid_to_idx = {int(c["id"]): i for i, c in enumerate(cats_sorted)}
    names = [str(c["name"]) for c in cats_sorted]
    nc = len(names)

    print(f"[coco_to_yolo_seg] nc={nc}")

    n1 = convert_split(Path(args.train_json), images_root, out_images / "train", out_labels / "train", catid_to_idx)
    n2 = convert_split(Path(args.val_json), images_root, out_images / "val", out_labels / "val", catid_to_idx)
    n3 = convert_split(Path(args.test_json), images_root, out_images / "test", out_labels / "test", catid_to_idx)

    # dataset.yaml
    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    meta = {
        "out_dir": str(out_dir),
        "images_root": str(images_root),
        "counts": {
            "train": {"images": n1[0], "polygons": n1[1]},
            "val": {"images": n2[0], "polygons": n2[1]},
            "test": {"images": n3[0], "polygons": n3[1]},
        },
        "nc": nc,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] dataset.yaml saved:", out_dir / "dataset.yaml")
    print("[OK] meta.json saved:", out_dir / "meta.json")


if __name__ == "__main__":
    main()