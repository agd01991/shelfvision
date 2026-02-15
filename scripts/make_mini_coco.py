import argparse
import json
import os
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        required=True,
        help="Папка data/raw/demo_coco (с images/ и annotations.json)",
    )
    ap.add_argument("--dst", required=True, help="Куда сохранить мини-датасет")
    ap.add_argument("--n", type=int, default=200, help="Сколько изображений выбрать")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    src_img = src / "images"
    src_ann = src / "annotations.json"

    dst_img = dst / "images"
    dst_img.mkdir(parents=True, exist_ok=True)

    coco = json.loads(src_ann.read_text(encoding="utf-8"))
    images = coco["images"]
    anns = coco["annotations"]
    cats = coco.get("categories", [])

    rnd = random.Random(args.seed)
    rnd.shuffle(images)
    pick = images[: min(args.n, len(images))]
    pick_ids = set(im["id"] for im in pick)

    # Фильтруем аннотации только по выбранным image_id
    pick_anns = [a for a in anns if a["image_id"] in pick_ids]

    # Копируем файлы и делаем file_name просто именем файла (без папок),
    # чтобы image_root мог быть data/raw/mini_coco/images
    fixed_images = []
    for im in pick:
        fname = Path(im["file_name"]).name
        src_path = src_img / fname
        if not src_path.exists():
            # на случай, если file_name уже просто имя
            src_path = src_img / im["file_name"]
        if not src_path.exists():
            continue

        shutil.copy2(src_path, dst_img / fname)
        im2 = dict(im)
        im2["file_name"] = fname
        fixed_images.append(im2)

    out = {
        "images": fixed_images,
        "annotations": pick_anns,
        "categories": cats,
    }
    (dst / "annotations.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved mini COCO to: {dst}")
    print(f"Images: {len(fixed_images)}, Anns: {len(pick_anns)}")


if __name__ == "__main__":
    main()
