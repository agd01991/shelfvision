from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

from src.core.config import ensure_dir, read_json, write_json, RunPaths
from src.core.timer import timed
from src.data.coco_schema import load_coco, dump_coco
from src.data.validators import validate_and_fix_bboxes
# from src.data.splits import SplitSpec, make_group_split
from src.data.splits import SplitSpec, make_split
from src.data.reports import draw_samples

import cv2

def sync_image_sizes_from_files(images, image_root: Path, logger, max_check: int = 0):
    """
    Обновляет width/height по реальным файлам.
    max_check=0 => проверять все. Иначе проверять первые N.
    """
    missing = 0
    checked = 0
    for im in images:
        if max_check and checked >= max_check:
            break
        checked += 1

        p = image_root / im.file_name
        if not p.exists():
            # fallback: иногда file_name только имя
            alt = image_root / Path(im.file_name).name
            if alt.exists():
                p = alt
            else:
                missing += 1
                continue

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            missing += 1
            continue

        h, w = img.shape[:2]
        if im.width != w or im.height != h:
            logger.warning(f"Size mismatch for {im.file_name}: coco=({im.width},{im.height}) real=({w},{h})")
            im.width = w
            im.height = h

    if missing:
        logger.warning(f"Missing/Unreadable images during size sync: {missing} (checked {checked})")

def build_run_paths(prepared_root: Path, version: str) -> RunPaths:
    version_dir = ensure_dir(prepared_root / version)
    images_dir = ensure_dir(version_dir / "images")
    reports_dir = ensure_dir(version_dir / "reports")
    return RunPaths(
        prepared_root=prepared_root,
        version_dir=version_dir,
        images_dir=images_dir,
        reports_dir=reports_dir,
        annotations_path=version_dir / "annotations.json",
        splits_path=version_dir / "splits.json",
        passport_path=version_dir / "passport.json",
        issues_path=version_dir / "issues.json",
    )


def prepare_from_coco_bbox(cfg: Dict[str, Any], version: str, logger) -> Tuple[RunPaths, Dict[str, float]]:
    timings: Dict[str, float] = {}
    source = cfg["source"]
    coco_json = Path(source["coco_json"])
    image_root = Path(source["image_root"])

    prepared_root = Path(cfg["prepared_dir"])
    paths = build_run_paths(prepared_root, version)

    with timed("load_coco", timings):
        coco_obj = read_json(coco_json)
        images, anns, cats = load_coco(coco_obj)
        sync_image_sizes_from_files(images, image_root, logger, max_check=0)

    vcfg = cfg.get("validation", {})
    min_area = float(vcfg.get("min_bbox_area", 16.0))
    clip_to_image = bool(vcfg.get("clip_to_image", True))

    with timed("validate", timings):
        anns_fixed, issues, stats = validate_and_fix_bboxes(
            images=images,
            anns=anns,
            min_area=min_area,
            clip_to_image=clip_to_image,
        )

    with timed("save_annotations", timings):
        write_json(paths.annotations_path, dump_coco(images, anns_fixed, cats))
        write_json(paths.issues_path, {
            "issues": issues,
            "stats": stats.__dict__,
            "min_area": min_area,
            "clip_to_image": clip_to_image,
        })

    scfg = cfg["split"]
    # spec = SplitSpec(
    #     train=float(scfg["train"]),
    #     val=float(scfg["val"]),
    #     test=float(scfg["test"]),
    #     seed=int(scfg.get("seed", 42)),
    # )
    spec = SplitSpec(
    train=float(scfg["train"]),
    val=float(scfg["val"]),
    test=float(scfg["test"]),
    seed=int(scfg.get("seed", 42)),
    mode=scfg.get("mode", "group_by_parent_dir"),
)

    with timed("split", timings):
        # split_obj = make_group_split(images, spec)
        split_obj = make_split(images, spec)
        write_json(paths.splits_path, split_obj)

    with timed("passport", timings):
        passport = {
            "version": version,
            "source": {
                "type": "coco_bbox",
                "coco_json": str(coco_json),
                "image_root": str(image_root),
            },
            "counts": {
                "images_total": len(images),
                "annotations_total": len(anns),
                "annotations_kept": len(anns_fixed),
            },
            "validation": {
                "min_bbox_area": min_area,
                "clip_to_image": clip_to_image,
                "stats": stats.__dict__,
            },
            "split_meta": split_obj["meta"],
        }
        write_json(paths.passport_path, passport)

    with timed("report", timings):
        anns_by_image: Dict[int, list] = {}
        for a in anns_fixed:
            anns_by_image.setdefault(a.image_id, []).append(a)

        rep_cfg = cfg.get("report", {})
        sample_per_split = int(rep_cfg.get("sample_per_split", 25))
        draw_max_boxes = int(rep_cfg.get("draw_max_boxes", 80))
        seed = int(spec.seed)

        img_map = {im.id: im for im in images}

        for split_name in ["train", "val", "test"]:
            ids = split_obj[split_name]
            split_images = [img_map[i] for i in ids if i in img_map]
            out_dir = paths.reports_dir / f"samples_{split_name}"
            draw_samples(
                image_root=image_root,
                images=split_images,
                anns_by_image=anns_by_image,
                out_dir=out_dir,
                sample_n=sample_per_split,
                draw_max_boxes=draw_max_boxes,
                seed=seed,
            )

    logger.info(f"Prepared dataset saved to: {paths.version_dir}")
    return paths, timings
