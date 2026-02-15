from __future__ import annotations
import argparse
from pathlib import Path
import sys

# чтобы `from src...` работало при запуске из корня проекта
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_yaml  # noqa: E402
from src.core.logging import setup_logging  # noqa: E402
from src.data.prepare import (  # noqa: E402
    prepare_from_coco_bbox,
    prepare_from_sku110k_bbox,
    prepare_from_d2s_instances,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/datasets.yaml")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()

    logger = setup_logging()

    cfg = load_yaml(args.config)
    if "datasets" not in cfg or args.dataset not in cfg["datasets"]:
        raise KeyError(f"Dataset '{args.dataset}' not found in {args.config}")

    ds_cfg = cfg["datasets"][args.dataset]
    source_type = ds_cfg["source"]["type"]

    if source_type == "coco_bbox":
        _, timings = prepare_from_coco_bbox(ds_cfg, args.version, logger)
        logger.info(f"Timings (ms): { {k: round(v, 2) for k, v in timings.items()} }")
    elif source_type == "sku110k_bbox":
        _, timings = prepare_from_sku110k_bbox(ds_cfg, args.version, logger)
        logger.info(f"Timings (ms): { {k: round(v, 2) for k, v in timings.items()} }")
    elif source_type == "d2s_instances":
        _, timings = prepare_from_d2s_instances(ds_cfg, args.version, logger)
        logger.info(f"Timings (ms): { {k: round(v, 2) for k, v in timings.items()} }")
    else:
        raise NotImplementedError(f"Source type '{source_type}' is not implemented yet")


if __name__ == "__main__":
    main()
