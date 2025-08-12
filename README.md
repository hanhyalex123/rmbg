# 一键抠图（RMBG-2.0）增强方案

本仓库提供两条一键抠图管线：

- 零训练版：仅使用 `briaai/RMBG-2.0` 推理，直接输出前景透明 PNG。
- 微训练版：在零训练版基础上加入“轻量精修器”（可选，5–10ms 级），或仅做引导滤波精修（无训练）。

## 环境

优先使用系统默认 Python；如需独立环境：

```bash
python -m venv venv_rmbg
source venv_rmbg/bin/activate
pip install -r requirements.txt
```

## 依赖

核心：

```text
torch
torchvision
transformers
Pillow
numpy
```

可选（微训练/精修）：

```text
opencv-python-headless
fastguidedfilter
pymatting
```

## 使用

1) 零训练版：

```bash
python -m pipelines.zero_shot <path_to_image> [output_dir]
```

2) 无训练微精修：

```bash
python -m pipelines.micro_refine <path_to_image> [output_dir]
```

3) 轻量精修器微训练（可选）：

```bash
python train_refiner.py --data_root /path/to/data --epochs 5
```

数据目录结构（示例）：

```
/path/to/data
  ├─ images/*.png|jpg
  └─ alphas/*.png   # 0..255 alpha GT
```

## API 服务（FastAPI）

提供 3 个接口：

- `POST /api/cutout/zero`：零训练一键（RMBG-2.0 直出）
  - 表单：`file`(UploadFile), `output_dir`(可选)
- `POST /api/cutout/micro`：无训练轻量精修（RMBG + 引导滤波，支持加载微训练精修器）
  - 表单：`file`, `output_dir`(可选), `refiner_ckpt`(可选)
- `POST /api/cutout/semantic`：语义融合（BiRefNet 前景 × RMBG alpha）
  - 表单：`file`, `output_dir`(可选), `birefnet_ckpt`(可选)

本地启动（不会自动启动服务，需要你手动执行）：

```bash
pip install fastapi uvicorn
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

健康检查：`GET /health`

返回字段：均返回 `saved_path` 和 `result_base64`，可直接显示 PNG。

## 参考

- RMBG 2.0: `https://huggingface.co/briaai/RMBG-2.0`
- BiRefNet（可选语义先验）: `https://github.com/ZhengPeng7/BiRefNet`


