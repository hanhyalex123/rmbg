import os
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- 1. 导入worker并初始化 ---
import worker

# 任务状态的内存数据库
# 格式: { "task_id": {"status": "...", "result": ...} }
TASK_DB: Dict[str, Dict[str, Any]] = {}

# 定义任务状态常量
class TaskStatus:
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# 确保输入和输出目录存在
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 2. 后台任务处理函数 ---
def run_processing_task(image_path: str, output_dir: Path, task_id: str, resolution: str | None, num_images: int,
                        guided: bool, prompt: str | None, negative_prompt: str | None):
    try:
        result_paths = worker.process_image(image_path, output_dir, task_id, resolution, num_images,
                                            guided=guided, prompt=prompt, negative_prompt=negative_prompt)
        TASK_DB[task_id] = {
            "status": TaskStatus.COMPLETED,
            "result": result_paths
        }
        print(f"任务 {task_id} 已成功完成。")
    except Exception as e:
        error_message = f"任务 {task_id} 处理失败: {str(e)}"
        print(error_message)
        TASK_DB[task_id] = {
            "status": TaskStatus.FAILED,
            "error": error_message
        }
    finally:
        try:
            os.remove(image_path)
            print(f"已清理输入文件: {image_path}")
        except OSError as e:
            print(f"清理文件 {image_path} 时出错: {e}")

# --- 新增：只抠前景后台任务 ---
def run_foreground_task(image_path: str, output_dir: Path, task_id: str):
    try:
        fg_path, mask_path = worker.extract_foreground(image_path, output_dir, task_id)
        TASK_DB[task_id] = {
            "status": TaskStatus.COMPLETED,
            "result": {
                "foreground_url": f"/outputs/{Path(fg_path).name}",
                "mask_url": f"/outputs/{Path(mask_path).name}"
            }
        }
        print(f"前景任务 {task_id} 已成功完成。")
    except Exception as e:
        error_message = f"前景任务 {task_id} 处理失败: {str(e)}"
        print(error_message)
        TASK_DB[task_id] = {
            "status": TaskStatus.FAILED,
            "error": error_message
        }
    finally:
        try:
            os.remove(image_path)
            print(f"已清理输入文件: {image_path}")
        except OSError as e:
            print(f"清理文件 {image_path} 时出错: {e}")

# --- 3. FastAPI 应用设置 ---
app = FastAPI(
    title="图像Inpainting服务",
    description="一个通过上传图片和taskId来异步处理图像并返回结果的API。",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    print("服务器启动... 正在预加载AI模型...")
    try:
        worker.load_models()
        print("所有模型已成功加载。服务准备就绪。")
    except Exception as e:
        print(f"致命错误：模型加载失败，服务将无法正常工作。错误: {e}")

class TaskCreationResponse(BaseModel):
    message: str = "任务已接收，正在后台处理"
    task_id: str

@app.post("/v1/images/generations", response_model=TaskCreationResponse, status_code=202)
async def create_generation_task(
    background_tasks: BackgroundTasks,
    taskId: str = Form(...),
    image: UploadFile = File(...),
    output_resolution: str | None = Form(None, description="期望的输出分辨率，格式为 '宽x高'。如果留空，将自动缩放至2K。"),
    num_images: int = Form(1, description="要生成的图片数量。", ge=1, le=8),
    guided: bool = Form(True, description="是否使用闭运算+guided filter 优化边缘。"),
    prompt: str | None = Form(None),
    negative_prompt: str | None = Form(None)
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件类型错误，请上传图片。")
    if taskId in TASK_DB:
         raise HTTPException(status_code=409, detail=f"任务ID '{taskId}' 已存在。请使用唯一的taskId。")
    file_extension = Path(image.filename).suffix or ".jpg"
    image_path = INPUT_DIR / f"{taskId}{file_extension}"
    try:
        with open(image_path, "wb") as f:
            f.write(await image.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"无法保存上传的文件: {e}")
    TASK_DB[taskId] = {"status": TaskStatus.PROCESSING}
    background_tasks.add_task(run_processing_task, str(image_path), OUTPUT_DIR, taskId, output_resolution, num_images,
                              guided, prompt, negative_prompt)
    return {"task_id": taskId}

# --- 新增：只抠前景接口 ---
@app.post("/v1/images/foreground", response_model=TaskCreationResponse, status_code=202)
async def create_foreground_task(
    background_tasks: BackgroundTasks,
    taskId: str = Form(...),
    image: UploadFile = File(...)
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件类型错误，请上传图片。")
    if taskId in TASK_DB:
         raise HTTPException(status_code=409, detail=f"任务ID '{taskId}' 已存在。请使用唯一的taskId。")
    file_extension = Path(image.filename).suffix or ".jpg"
    image_path = INPUT_DIR / f"{taskId}{file_extension}"
    try:
        with open(image_path, "wb") as f:
            f.write(await image.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"无法保存上传的文件: {e}")
    TASK_DB[taskId] = {"status": TaskStatus.PROCESSING}
    background_tasks.add_task(run_foreground_task, str(image_path), OUTPUT_DIR, taskId)
    return {"task_id": taskId}

@app.get("/v1/images/results/{task_id}")
async def get_task_result(task_id: str):
    task_result = TASK_DB.get(task_id)
    if task_result is None:
        raise HTTPException(status_code=404, detail=f"任务ID '{task_id}' 未找到。")
    return JSONResponse(content=task_result)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "服务已启动。请访问 /docs 查看API文档。"} 

# 健康检查
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}

@app.get("/readyz", include_in_schema=False)
def readyz():
    # 简化判断：模型是否已加载
    try:
        worker.load_models()
        return {"ready": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ready": False, "error": str(e)})