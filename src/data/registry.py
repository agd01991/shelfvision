# from __future__ import annotations
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, Literal

# from src.core.config import load_yaml


# SourceType = Literal["coco_bbox"]


# @dataclass(frozen=True)
# class DatasetSpec:
#     name: str
#     raw_dir: Path
#     prepared_dir: Path
#     source_type: SourceType
#     source: Dict[str, Any]
#     split: Dict[str, Any]
#     validation: Dict[str, Any]
#     report: Dict[str, Any]


# class DatasetRegistry:
#     def __init__(self, yaml_path: str | Path):
#         cfg = load_yaml(yaml_path)
#         self._items: Dict[str, DatasetSpec] = {}
#         for name, spec in cfg.get("datasets", {}).items():
#             self._items[name] = DatasetSpec(
#                 name=name,
#                 raw_dir=Path(spec["raw_dir"]),
#                 prepared_dir=Path(spec["prepared_dir"]),
#                 source_type=spec["source"]["type"],
#                 source=spec["source"],
#                 split=spec["split"],
#                 validation=spec.get("validation", {}),
#                 report=spec.get("report", {}),
#             )

#     def get(self, name: str) -> DatasetSpec:
#         if name not in self._items:
#             raise KeyError(f"Unknown dataset: {name}")
#         return self._items[name]

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from src.core.config import load_yaml

AnnoType = Literal["bbox", "mask", "bbox+sku", "mask+sku"]

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_dir: Path
    prepared_dir: Path
    anno_type: AnnoType
    image_glob: str
    split_mode: str
    tile: dict

class DatasetRegistry:
    def __init__(self, yaml_path: str | Path):
        cfg = load_yaml(yaml_path)
        self._items = {}
        for name, spec in cfg["datasets"].items():
            self._items[name] = DatasetSpec(
                name=name,
                raw_dir=Path(spec["raw_dir"]),
                prepared_dir=Path(spec["prepared_dir"]),
                anno_type=spec["anno_type"],
                image_glob=spec.get("image_glob", "**/*.jpg"),
                split_mode=spec.get("split_mode", "group"),
                tile=spec.get("tile", {"enabled": False}),
            )

    def get(self, name: str) -> DatasetSpec:
        if name not in self._items:
            raise KeyError(f"Unknown dataset: {name}")
        return self._items[name]
