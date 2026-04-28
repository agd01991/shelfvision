from __future__ import annotations

import argparse
import json
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--out_dir", default="runs/d2s_maskrcnn")
    ap.add_argument("--max_iter", type=int, default=15000)
    ap.add_argument("--ims_per_batch", type=int, default=2)
    ap.add_argument("--base_lr", type=float, default=0.00025)
    ap.add_argument("--min_size", type=int, default=640)
    ap.add_argument("--max_size", type=int, default=640)

    # NEW:
    ap.add_argument("--weights_path", default="", help="local .pkl/.pth to avoid downloading")
    ap.add_argument("--num_classes", type=int, default=0, help="0=auto from train_json categories")
    args = ap.parse_args()

    images_dir = str(Path(args.images_dir).resolve())
    train_json = Path(args.train_json).resolve()
    val_json = str(Path(args.val_json).resolve())

    register_coco_instances("d2s_train", {}, str(train_json), images_dir)
    register_coco_instances("d2s_val", {}, val_json, images_dir)

    # auto num_classes from COCO categories
    coco_train = json.loads(train_json.read_text(encoding="utf-8"))
    auto_classes = len(coco_train.get("categories", []))
    num_classes = args.num_classes if args.num_classes > 0 else auto_classes
    if num_classes <= 0:
        raise ValueError("Cannot determine num_classes. Check train.json categories.")

    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.INPUT.MASK_FORMAT = "bitmask"

    if args.weights_path:
        cfg.MODEL.WEIGHTS = str(Path(args.weights_path).expanduser().resolve())
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("d2s_train",)
    cfg.DATASETS.TEST = ("d2s_val",)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.AMP.ENABLED = True

    cfg.INPUT.MIN_SIZE_TRAIN = (args.min_size,)
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size
    cfg.INPUT.MIN_SIZE_TEST = args.min_size
    cfg.INPUT.MAX_SIZE_TEST = args.max_size

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = str(Path(args.out_dir).resolve())
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()