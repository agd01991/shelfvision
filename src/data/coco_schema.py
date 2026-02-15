from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, w, h]
    area: Optional[float] = None
    iscrowd: int = 0
    segmentation: Optional[Any] = None


@dataclass
class CocoCategory:
    id: int
    name: str


def load_coco(
    obj: Dict[str, Any],
) -> Tuple[List[CocoImage], List[CocoAnnotation], List[CocoCategory]]:
    images: List[CocoImage] = []
    for im in obj.get("images", []):
        images.append(
            CocoImage(
                id=int(im["id"]),
                file_name=str(im["file_name"]),
                width=int(im["width"]),
                height=int(im["height"]),
            )
        )

    anns: List[CocoAnnotation] = []
    for a in obj.get("annotations", []):
        bbox = a.get("bbox", [0, 0, 0, 0])
        anns.append(
            CocoAnnotation(
                id=int(a["id"]),
                image_id=int(a["image_id"]),
                category_id=int(a["category_id"]),
                bbox=[float(x) for x in bbox],
                area=float(a["area"])
                if "area" in a and a["area"] is not None
                else None,
                iscrowd=int(a.get("iscrowd", 0)),
                segmentation=a.get("segmentation", None),
            )
        )

    cats: List[CocoCategory] = []
    for c in obj.get("categories", []):
        cats.append(CocoCategory(id=int(c["id"]), name=str(c["name"])))

    return images, anns, cats


def dump_coco(
    images: List[CocoImage], anns: List[CocoAnnotation], cats: List[CocoCategory]
) -> Dict[str, Any]:
    return {
        "images": [im.__dict__ for im in images],
        "annotations": [a.__dict__ for a in anns],
        "categories": [c.__dict__ for c in cats],
    }
