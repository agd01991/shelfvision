from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

from src.core.config import RunPaths, ensure_dir, read_json, write_json
from src.core.timer import timed
from src.data.coco_schema import (
    CocoAnnotation,
    CocoCategory,
    CocoImage,
    dump_coco,
    load_coco,
)
from src.data.d2s_reader import load_d2s_as_coco_from_instance_id_png
from src.data.reports import draw_samples
from src.data.sku110k_reader import load_sku110k_as_coco
from src.data.splits import SplitSpec, make_split
from src.data.tiling import TileConfig, tile_coco
from src.data.validators import validate_and_fix_bboxes


# -------------------------
# Helpers
# -------------------------


def sync_image_sizes_from_files(
    images: List[CocoImage], image_root: Path, logger, max_check: int = 0
) -> None:

    image_root = Path(image_root)
    checked = 0
    missing_or_bad = 0
    fixed = 0

    for im in images:
        if max_check and checked >= max_check:
            break
        checked += 1

        p = image_root / im.file_name
        if not p.exists():
            # sometimes COCO stores nested paths, but files are flat
            alt = image_root / Path(im.file_name).name
            if alt.exists():
                p = alt
            else:
                missing_or_bad += 1
                continue

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            missing_or_bad += 1
            continue

        h, w = img.shape[:2]
        if im.width != w or im.height != h:
            logger.warning(
                f"[sync_sizes] {im.file_name}: coco=({im.width},{im.height}) real=({w},{h})"
            )
            im.width = int(w)
            im.height = int(h)
            fixed += 1

    logger.info(
        f"[sync_sizes] checked={checked}, fixed={fixed}, missing_or_bad={missing_or_bad}"
    )


def build_run_paths(prepared_root: Path, version: str) -> RunPaths:
    """Create target directories and file paths for a prepared version."""
    prepared_root = Path(prepared_root)
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


def _split_map_from_obj(split_obj: Dict[str, Any]) -> Dict[int, str]:
    """Convert splits.json object to {image_id: split_name}."""
    m: Dict[int, str] = {}
    for s in ("train", "val", "test"):
        for image_id in split_obj.get(s, []):
            m[int(image_id)] = s
    return m


def _split_obj_from_map(
    split_map: Dict[int, str], seed: int, mode: str = "inherit"
) -> Dict[str, Any]:
    """Convert {image_id: split_name} back to splits.json structure."""
    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for image_id, s in split_map.items():
        if s == "val":
            val.append(int(image_id))
        elif s == "test":
            test.append(int(image_id))
        else:
            train.append(int(image_id))

    train.sort()
    val.sort()
    test.sort()

    return {
        "train": train,
        "val": val,
        "test": test,
        "meta": {
            "mode": mode,
            "images_total": len(split_map),
            "seed": int(seed),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
    }


# -------------------------
# Common pipeline
# -------------------------


def _prepare_common(
    *,
    cfg: Dict[str, Any],
    version: str,
    logger,
    images: List[CocoImage],
    anns: List[CocoAnnotation],
    cats: List[CocoCategory],
    image_root: Path,
    source_meta: Dict[str, Any],
) -> Tuple[RunPaths, Dict[str, float]]:
    """Common preparation pipeline.

    The source loader is responsible for producing (images, anns, cats, image_root).
    Everything else (validation/split/tiling/save/reports) is handled here.
    """

    timings: Dict[str, float] = {}
    prepared_root = Path(cfg["prepared_dir"])
    paths = build_run_paths(prepared_root, version)

    # 0) optional size sync
    sync_cfg = cfg.get("sync_sizes", {})
    do_sync = bool(sync_cfg.get("enabled", True))
    sync_max_check = int(sync_cfg.get("max_check", 0))

    if do_sync:
        with timed("sync_sizes", timings):
            sync_image_sizes_from_files(
                images, image_root, logger, max_check=sync_max_check
            )

    # 1) validate/fix bboxes
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

    # 2) split
    scfg = cfg["split"]
    spec = SplitSpec(
        train=float(scfg["train"]),
        val=float(scfg["val"]),
        test=float(scfg["test"]),
        seed=int(scfg.get("seed", 42)),
        mode=str(scfg.get("mode", "random_by_image")),
    )

    with timed("split", timings):
        split_obj = make_split(images, spec)

    # 3) optional tiling (part of prepared)
    tiling_cfg = cfg.get("tiling", {})
    tiling_enabled = bool(tiling_cfg.get("enabled", False))
    tile_map: Dict[int, Dict[str, Any]] | None = None

    if tiling_enabled:
        with timed("tiling", timings):
            tc = TileConfig(
                enabled=True,
                tile_size=int(tiling_cfg.get("tile_size", 1024)),
                overlap=float(tiling_cfg.get("overlap", 0.2)),
                min_visible=float(tiling_cfg.get("min_visible", 0.25)),
                save_format=str(tiling_cfg.get("save_format", "jpg")),
                jpeg_quality=int(tiling_cfg.get("jpeg_quality", 92)),
            )

            split_map = _split_map_from_obj(split_obj)
            tile_images_dir = paths.images_dir / "tiles"
            tile_images, tile_anns, tile_map, tile_split_map = tile_coco(
                images=images,
                anns=anns_fixed,
                image_root=image_root,
                out_images_dir=tile_images_dir,
                split_map=split_map,
                cfg=tc,
                logger=logger,
            )

            # Switch the pipeline to work on tiles
            images = tile_images
            anns_fixed = tile_anns
            image_root = tile_images_dir
            split_obj = _split_obj_from_map(
                tile_split_map, seed=spec.seed, mode="inherit_from_source"
            )

            # Persist tile mapping
            write_json(paths.version_dir / "tile_map.json", tile_map)

    # 4) save artifacts
    with timed("save", timings):
        write_json(paths.annotations_path, dump_coco(images, anns_fixed, cats))
        write_json(paths.splits_path, split_obj)
        write_json(
            paths.issues_path,
            {
                "source": source_meta,
                "validation": {
                    "min_bbox_area": min_area,
                    "clip_to_image": clip_to_image,
                    "stats": stats.__dict__,
                },
                "issues": issues,
            },
        )

        passport: Dict[str, Any] = {
            "version": version,
            "source": source_meta,
            "counts": {
                # source counts (before validation)
                "annotations_total": len(anns),
                # prepared counts (after validation / tiling)
                "images_total": len(images),
                "annotations_kept": len(anns_fixed),
            },
            "validation": {
                "min_bbox_area": min_area,
                "clip_to_image": clip_to_image,
                "stats": stats.__dict__,
            },
            "split_meta": split_obj.get("meta", {}),
        }

        if do_sync:
            passport["sync_sizes"] = {"enabled": True, "max_check": sync_max_check}

        if tiling_enabled:
            passport["tiling"] = {
                "enabled": True,
                "tile_size": int(tiling_cfg.get("tile_size", 1024)),
                "overlap": float(tiling_cfg.get("overlap", 0.2)),
                "min_visible": float(tiling_cfg.get("min_visible", 0.25)),
                "images_dir": str((paths.images_dir / "tiles").as_posix()),
                "tile_map": str((paths.version_dir / "tile_map.json").as_posix()),
            }

        write_json(paths.passport_path, passport)

    # 5) visual reports (samples)
    with timed("report", timings):
        anns_by_image: Dict[int, List[CocoAnnotation]] = {}
        for a in anns_fixed:
            anns_by_image.setdefault(a.image_id, []).append(a)

        rep_cfg = cfg.get("report", {})
        sample_per_split = int(rep_cfg.get("sample_per_split", 25))
        draw_max_boxes = int(rep_cfg.get("draw_max_boxes", 80))

        img_map = {im.id: im for im in images}
        for split_name in ("train", "val", "test"):
            ids = split_obj.get(split_name, [])
            split_images = [img_map[i] for i in ids if i in img_map]
            out_dir = paths.reports_dir / f"samples_{split_name}"
            draw_samples(
                image_root=image_root,
                images=split_images,
                anns_by_image=anns_by_image,
                out_dir=out_dir,
                sample_n=sample_per_split,
                draw_max_boxes=draw_max_boxes,
                seed=int(spec.seed),
            )

    logger.info(f"Prepared dataset saved: {paths.version_dir}")
    return paths, timings


# -------------------------
# Source-specific entry points
# -------------------------


def prepare_from_coco_bbox(
    cfg: Dict[str, Any], version: str, logger
) -> Tuple[RunPaths, Dict[str, float]]:
    """Prepare dataset from an already COCO-bbox source."""
    source = cfg["source"]
    coco_json = Path(source["coco_json"])
    image_root = Path(source["image_root"])

    coco_obj = read_json(coco_json)
    images, anns, cats = load_coco(coco_obj)

    return _prepare_common(
        cfg=cfg,
        version=version,
        logger=logger,
        images=images,
        anns=anns,
        cats=cats,
        image_root=image_root,
        source_meta={
            "type": "coco_bbox",
            "coco_json": str(coco_json),
            "image_root": str(image_root),
        },
    )


def prepare_from_sku110k_bbox(
    cfg: Dict[str, Any], version: str, logger
) -> Tuple[RunPaths, Dict[str, float]]:
    """Prepare SKU-110K CSV annotations as COCO-bbox."""
    src = cfg["source"]
    images_dir = Path(src["images_dir"])
    csv_path = Path(src["csv_path"])
    category_name = str(src.get("category_name", "product"))
    category_id = int(src.get("category_id", 1))

    images, anns, cats = load_sku110k_as_coco(
        images_dir=images_dir,
        csv_path=csv_path,
        category_name=category_name,
        category_id=category_id,
        logger=logger,
    )

    return _prepare_common(
        cfg=cfg,
        version=version,
        logger=logger,
        images=images,
        anns=anns,
        cats=cats,
        image_root=images_dir,
        source_meta={
            "type": "sku110k_bbox",
            "images_dir": str(images_dir),
            "csv_path": str(csv_path),
            "category_name": category_name,
            "category_id": category_id,
        },
    )


def prepare_from_d2s_instances(
    cfg: Dict[str, Any], version: str, logger
) -> Tuple[RunPaths, Dict[str, float]]:
    """Prepare D2S instance-id masks into COCO (bbox + segmentation as RLE)."""
    src = cfg["source"]
    images_dir = Path(src["images_dir"])
    masks_dir = Path(src["masks_dir"])
    encoding = str(src.get("encoding", "class_times_1000_plus_instance"))

    cats: List[CocoCategory] = []
    for c in src.get("categories", []):
        cats.append(CocoCategory(id=int(c["id"]), name=str(c["name"])))

    images, anns, cats = load_d2s_as_coco_from_instance_id_png(
        images_dir=images_dir,
        masks_dir=masks_dir,
        encoding=encoding,
        categories=cats,
        mask_suffix=str(src.get("mask_suffix", ".png")),
        logger=logger,
    )

    return _prepare_common(
        cfg=cfg,
        version=version,
        logger=logger,
        images=images,
        anns=anns,
        cats=cats,
        image_root=images_dir,
        source_meta={
            "type": "d2s_instances",
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "encoding": encoding,
        },
    )
