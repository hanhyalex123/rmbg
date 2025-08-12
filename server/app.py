import base64
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from pipelines.zero_shot import run_zero_shot
from pipelines.micro_refine import run_micro_refine
from pipelines.semantic_fuse import run_semantic_fuse


app = FastAPI(title="RMBG One-Click Cutout Service")


def _to_b64(png_path: str) -> str:
    with open(png_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"service": "rmbg", "apis": ["/api/cutout/zero", "/api/cutout/micro", "/api/cutout/semantic"]}


@app.post("/api/cutout/zero")
async def api_zero(file: UploadFile = File(...), output_dir: Optional[str] = Form("outputs")):
    content = await file.read()
    tmp_dir = Path(".tmp"); tmp_dir.mkdir(exist_ok=True)
    tmp_path = str(tmp_dir / file.filename)
    with open(tmp_path, "wb") as f:
        f.write(content)
    out_path = run_zero_shot(tmp_path, output_dir)
    b64 = _to_b64(out_path)
    return JSONResponse({"mode": "zero", "saved_path": out_path, "result_base64": b64})


@app.post("/api/cutout/micro")
async def api_micro(
    file: UploadFile = File(...),
    output_dir: Optional[str] = Form("outputs"),
    refiner_ckpt: Optional[str] = Form(None),
):
    content = await file.read()
    tmp_dir = Path(".tmp"); tmp_dir.mkdir(exist_ok=True)
    tmp_path = str(tmp_dir / file.filename)
    with open(tmp_path, "wb") as f:
        f.write(content)
    out_path = run_micro_refine(tmp_path, output_dir, refiner_ckpt)
    b64 = _to_b64(out_path)
    return JSONResponse({"mode": "micro", "saved_path": out_path, "result_base64": b64})


@app.post("/api/cutout/semantic")
async def api_semantic(
    file: UploadFile = File(...),
    output_dir: Optional[str] = Form("outputs"),
    birefnet_ckpt: Optional[str] = Form(None),
):
    content = await file.read()
    tmp_dir = Path(".tmp"); tmp_dir.mkdir(exist_ok=True)
    tmp_path = str(tmp_dir / file.filename)
    with open(tmp_path, "wb") as f:
        f.write(content)
    out_path = run_semantic_fuse(tmp_path, output_dir, birefnet_ckpt)
    b64 = _to_b64(out_path)
    return JSONResponse({"mode": "semantic", "saved_path": out_path, "result_base64": b64})


