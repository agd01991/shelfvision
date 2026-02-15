# src/models/detectron/train.py
from pathlib import Path
from src.core.config import load_yaml, ensure_dir
from src.core.seed import seed_all


def train_detectron(cfg_path: str | Path) -> Path:
    cfg = load_yaml(cfg_path)
    seed_all(int(cfg.get("seed", 42)))

    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances

    run_dir = ensure_dir(cfg["run_dir"])
    ds_name = cfg["dataset_name"]
    register_coco_instances(ds_name, {}, cfg["coco_json"], cfg["image_root"])

    dcfg = get_cfg()
    dcfg.merge_from_file(cfg["model_zoo_config"])
    dcfg.OUTPUT_DIR = str(run_dir)
    dcfg.SOLVER.BASE_LR = float(cfg["lr"])
    dcfg.SOLVER.MAX_ITER = int(cfg["max_iter"])
    dcfg.SOLVER.IMS_PER_BATCH = int(cfg["ims_per_batch"])
    dcfg.DATASETS.TRAIN = (ds_name,)
    dcfg.DATASETS.TEST = ()

    trainer = DefaultTrainer(dcfg)
    trainer.resume_or_load(resume=bool(cfg.get("resume", False)))
    trainer.train()
    return run_dir
