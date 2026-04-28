from __future__ import annotations

import argparse
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
    ap.add_argument("--tiles_dir", required=True)  # папка с изображениями tiles
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--out_dir", default="runs/dir1_detectron_frcnn")
    ap.add_argument("--max_iter", type=int, default=15000)
    ap.add_argument("--ims_per_batch", type=int, default=2)
    ap.add_argument("--base_lr", type=float, default=0.00025)
    args = ap.parse_args()

    tiles = str(Path(args.tiles_dir).resolve())
    train_json = str(Path(args.train_json).resolve())
    val_json = str(Path(args.val_json).resolve())

    register_coco_instances("shelf_train", {}, train_json, tiles)
    register_coco_instances("shelf_val", {}, val_json, tiles)

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 640

    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # )

    cfg.MODEL.WEIGHTS = "/mnt/c/Users/agd01/Downloads/model_final_280758.pkl"

    cfg.DATASETS.TRAIN = ("shelf_train",)
    cfg.DATASETS.TEST = ("shelf_val",)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []  # без step decay для простоты baseline
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # один класс "product"

    cfg.OUTPUT_DIR = str(Path(args.out_dir).resolve())
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
