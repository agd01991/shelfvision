# from __future__ import annotations
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List
# import random

# from src.data.coco_schema import CocoImage


# @dataclass(frozen=True)
# class SplitSpec:
#     train: float
#     val: float
#     test: float
#     seed: int


# def group_key_by_parent(file_name: str) -> str:
#     p = Path(file_name)
#     return str(p.parent).replace("\\", "/")


# def make_group_split(images: List[CocoImage], spec: SplitSpec) -> Dict[str, List[int] | Dict]:
#     groups: Dict[str, List[int]] = {}
#     for im in images:
#         g = group_key_by_parent(im.file_name)
#         groups.setdefault(g, []).append(im.id)

#     keys = list(groups.keys())
#     rnd = random.Random(spec.seed)
#     rnd.shuffle(keys)

#     n = len(keys)
#     n_train = int(round(n * spec.train))
#     n_val = int(round(n * spec.val))

#     train_keys = keys[:n_train]
#     val_keys = keys[n_train:n_train + n_val]
#     test_keys = keys[n_train + n_val:]

#     def collect(ks: List[str]) -> List[int]:
#         out: List[int] = []
#         for k in ks:
#             out.extend(groups[k])
#         return out

#     return {
#         "train": collect(train_keys),
#         "val": collect(val_keys),
#         "test": collect(test_keys),
#         "meta": {
#             "mode": "group_by_parent_dir",
#             "groups_total": n,
#             "train_groups": len(train_keys),
#             "val_groups": len(val_keys),
#             "test_groups": len(test_keys),
#             "seed": spec.seed,
#         },
#     }

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal
import random

from src.data.coco_schema import CocoImage

SplitMode = Literal["group_by_parent_dir", "random_by_image", "hash_by_image_id"]


@dataclass(frozen=True)
class SplitSpec:
    train: float
    val: float
    test: float
    seed: int
    mode: SplitMode


def group_key_by_parent(file_name: str) -> str:
    p = Path(file_name)
    return str(p.parent).replace("\\", "/")


def make_split(images: List[CocoImage], spec: SplitSpec) -> Dict[str, List[int]]:
    if spec.mode == "group_by_parent_dir":
        return _make_group_split(images, spec)
    if spec.mode == "random_by_image":
        return _make_random_split(images, spec)
    if spec.mode == "hash_by_image_id":
        return _make_hash_split(images, spec)
    raise ValueError(f"Unknown split mode: {spec.mode}")


def _make_random_split(images: List[CocoImage], spec: SplitSpec) -> Dict[str, List[int]]:
    ids = [im.id for im in images]
    rnd = random.Random(spec.seed)
    rnd.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * spec.train))
    n_val = int(round(n * spec.val))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "meta": {
            "mode": "random_by_image",
            "images_total": n,
            "seed": spec.seed,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }


def _make_hash_split(images: List[CocoImage], spec: SplitSpec) -> Dict[str, List[int]]:
    # Детерминированный сплит без RNG: по остатку от деления image_id
    train_ids, val_ids, test_ids = [], [], []
    for im in images:
        r = im.id % 10
        if r < int(spec.train * 10):
            train_ids.append(im.id)
        elif r < int((spec.train + spec.val) * 10):
            val_ids.append(im.id)
        else:
            test_ids.append(im.id)

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "meta": {
            "mode": "hash_by_image_id",
            "images_total": len(images),
            "seed": spec.seed,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }


def _make_group_split(images: List[CocoImage], spec: SplitSpec) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for im in images:
        g = group_key_by_parent(im.file_name)
        groups.setdefault(g, []).append(im.id)

    keys = list(groups.keys())
    rnd = random.Random(spec.seed)
    rnd.shuffle(keys)

    n = len(keys)
    n_train = int(round(n * spec.train))
    n_val = int(round(n * spec.val))
    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    test_keys = keys[n_train + n_val:]

    def collect(ks: List[str]) -> List[int]:
        out: List[int] = []
        for k in ks:
            out.extend(groups[k])
        return out

    return {
        "train": collect(train_keys),
        "val": collect(val_keys),
        "test": collect(test_keys),
        "meta": {
            "mode": "group_by_parent_dir",
            "groups_total": n,
            "train_groups": len(train_keys),
            "val_groups": len(val_keys),
            "test_groups": len(test_keys),
            "seed": spec.seed,
        },
    }

