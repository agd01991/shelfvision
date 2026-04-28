from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_best_dir(root: Path, name: str, max_depth: int = 3) -> Path:
    """Ищет папку с нужным именем (images/annotations) и берёт самую 'богатую' по файлам."""
    best = None
    best_count = -1
    for p in root.rglob(name):
        if not p.is_dir():
            continue
        # ограничим глубину, чтобы не уехать далеко
        try:
            if len(p.relative_to(root).parts) > max_depth:
                continue
        except Exception:
            pass

        if name == "images":
            cnt = sum(1 for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS)
        else:
            cnt = sum(1 for x in p.rglob("*.json") if x.is_file())
        if cnt > best_count:
            best = p
            best_count = cnt

    if best is None:
        raise FileNotFoundError(f"Cannot find '{name}' dir under {root}")
    return best.resolve()


def pick_coco_jsons(ann_dir: Path) -> Tuple[Path, Path | None]:
    """Авто-выбор COCO json: если есть train+val — берём оба, иначе один файл."""
    jsons = sorted([p for p in ann_dir.rglob("*.json") if p.is_file()], key=lambda p: p.stat().st_size, reverse=True)
    if not jsons:
        raise FileNotFoundError(f"No json files in {ann_dir}")

    # приоритет: файлы с 'trainval'/'train_val'/'all'
    for p in jsons:
        n = p.name.lower()
        if "trainval" in n or "train_val" in n or "all" in n:
            return p, None

    trains = [p for p in jsons if "train" in p.name.lower()]
    vals = [p for p in jsons if ("val" in p.name.lower()) or ("valid" in p.name.lower())]

    train_json = trains[0] if trains else jsons[0]
    val_json = vals[0] if vals else None
    return train_json, val_json


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_file_name(fn: str) -> str:
    fn = fn.replace("\\", "/").lstrip("/")
    # если внутри file_name уже есть "images/..." — убираем, потому что image_root будет .../images
    if fn.startswith("images/"):
        fn = fn[len("images/") :]
    return fn


def index_images(images_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Возвращает:
      rel_map: "relative/path.jpg" -> абсолютный Path
      base_map: "path.jpg" -> абсолютный Path (fallback, если в COCO file_name только basename)
    """
    rel_map: Dict[str, Path] = {}
    base_map: Dict[str, Path] = {}

    for p in images_dir.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(images_dir).as_posix()
        rel_map[rel] = p

        # basename fallback (если дубликаты имён — оставляем первый попавшийся)
        base_map.setdefault(p.name, p)

    return rel_map, base_map



def merge_coco(train: dict, val: dict | None) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Возвращает объединённые (images, annotations, categories).
    image_id и ann_id будут переиндексированы позже.
    """
    cats = train.get("categories", [])
    if val and val.get("categories"):
        # если одинаковые id/name — оставляем train, иначе можно было бы объединять, но D2S обычно одинаковый
        pass

    images = list(train.get("images", []))
    anns = list(train.get("annotations", []))

    if val:
        # добавляем уникальные по file_name
        existing = {normalize_file_name(im["file_name"]) for im in images}
        val_images = []
        for im in val.get("images", []):
            fn = normalize_file_name(im["file_name"])
            if fn not in existing:
                val_images.append(im)
        images.extend(val_images)

        # аннотации вал добавляем все (они ссылаются на image_id; потом переиндексируем по file_name)
        anns.extend(list(val.get("annotations", [])))

    # нормализуем file_name у всех images
    for im in images:
        im["file_name"] = normalize_file_name(im["file_name"])

    return images, anns, cats


def build_small_subset(
    images_all: List[dict],
    anns_all: List[dict],
    cats: List[dict],
    images_dir: Path,
    dst_root: Path,
    n: int,
    seed: int,
) -> None:
    random.seed(seed)

    dst_images = dst_root / "images"
    dst_root.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    # индексы аннотаций по file_name через image_id->file_name
    img_by_id = {int(im["id"]): im for im in images_all if "id" in im}
    fn_by_old_id: Dict[int, str] = {}
    for old_id, im in img_by_id.items():
        fn_by_old_id[old_id] = normalize_file_name(im["file_name"])

    anns_by_fn: Dict[str, List[dict]] = {}
    for a in anns_all:
        iid = int(a["image_id"])
        fn = fn_by_old_id.get(iid)
        if fn is None:
            continue
        anns_by_fn.setdefault(fn, []).append(a)

    # пул файлов
    all_fns = sorted({normalize_file_name(im["file_name"]) for im in images_all})
    if len(all_fns) < n:
        raise ValueError(f"Not enough images in pool: have {len(all_fns)}, need {n}")

    chosen = random.sample(all_fns, n)

    # индекс файлов на диске
    rel_map, base_map = index_images(images_dir)

    # новые COCO структуры
    new_images = []
    new_anns = []
    new_img_id = 1
    new_ann_id = 1

    for fn in chosen:
        src = rel_map.get(fn)
        if src is None:
            # fallback по basename
            src = base_map.get(Path(fn).name)
        if src is None or not src.exists():
            # пропускаем, но стараемся сохранить n → для простоты бросаем ошибку, чтобы стало ясно что искать
            raise FileNotFoundError(f"Image file not found for file_name='{fn}' in {images_dir}")

        # копирование (без хардлинков — максимально совместимо)
        dst = dst_images / fn
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

        # width/height берём из исходного image record, если есть
        w = h = 0
        # найдём исходный record по file_name
        im_rec = next((im for im in images_all if normalize_file_name(im["file_name"]) == fn), None)
        if im_rec:
            w = int(im_rec.get("width", 0))
            h = int(im_rec.get("height", 0))

        new_images.append({"id": new_img_id, "file_name": fn, "width": w, "height": h})

        # аннотации (bbox + segmentation сохраняются как есть)
        for a in anns_by_fn.get(fn, []):
            na = dict(a)
            na["id"] = new_ann_id
            na["image_id"] = new_img_id
            new_anns.append(na)
            new_ann_id += 1

        new_img_id += 1

    out = {
        "info": {"description": "D2S small subset"},
        "licenses": [],
        "images": new_images,
        "annotations": new_anns,
        "categories": cats,
    }

    (dst_root / "annotations.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (dst_root / "meta.json").write_text(
        json.dumps(
            {
                "source_images_dir": str(images_dir),
                "n": n,
                "seed": seed,
                "images_kept": len(new_images),
                "annotations_kept": len(new_anns),
                "categories": len(cats),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="root where d2s_full is extracted")
    ap.add_argument("--dst_root", default="data/raw/d2s_small", help="where to build small dataset")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_json", default="", help="optional path to train coco json")
    ap.add_argument("--val_json", default="", help="optional path to val coco json (optional)")

    args = ap.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()

    images_dir = find_best_dir(src_root, "images")
    ann_dir = find_best_dir(src_root, "annotations")

    print("[D2S] images_dir:", images_dir)
    print("[D2S] ann_dir   :", ann_dir)

    if args.train_json:
        train_json = Path(args.train_json).expanduser().resolve()
        val_json = Path(args.val_json).expanduser().resolve() if args.val_json else None
    else:
        train_json, val_json = pick_coco_jsons(ann_dir)

    print("[D2S] train_json:", train_json)
    print("[D2S] val_json  :", val_json)

    train = load_coco(train_json)
    val = load_coco(val_json) if val_json else None

    images_all, anns_all, cats = merge_coco(train, val)

    print(f"[D2S] pool images={len(images_all)}, anns={len(anns_all)}, cats={len(cats)}")
    build_small_subset(
        images_all=images_all,
        anns_all=anns_all,
        cats=cats,
        images_dir=images_dir,
        dst_root=dst_root,
        n=args.n,
        seed=args.seed,
    )
    print("[OK] d2s_small создан:", dst_root)


if __name__ == "__main__":
    main()  