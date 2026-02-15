# src/integrations/api.py
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import uuid

from src.infer.engine import InferenceEngine
from src.core.config import ensure_dir


def create_app(engine: InferenceEngine, storage_dir: str) -> FastAPI:
    app = FastAPI(title="ShelfVision API")
    storage = ensure_dir(storage_dir)

    @app.post("/analyze")
    async def analyze(store_id: str, zone_id: str, file: UploadFile = File(...)):
        job_id = str(uuid.uuid4())
        img_path = storage / f"{job_id}_{file.filename}"
        content = await file.read()
        img_path.write_bytes(content)

        meta = {"store_id": store_id, "zone_id": zone_id, "job_id": job_id}
        res = engine.run_image(img_path, meta)
        return {"job_id": job_id, "timings_ms": res.timings_ms, "report": res.report}

    return app
