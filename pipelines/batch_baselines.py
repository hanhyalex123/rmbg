import os
import csv
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from models.rmbg2 import RMBG2, rgba_compose
from models.birefnet import BiRefNet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_dir: str) -> List[str]:
    paths: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        p = os.path.join(input_dir, name)
        if os.path.isfile(p) and Path(p).suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    return paths


def run_rmbg(image_path: str, rmbg: RMBG2) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    alpha = rmbg.predict_alpha(image)
    rgba = rgba_compose(image, alpha)
    return rgba


def run_birefnet(
    image_path: str,
    br: BiRefNet,
    thr: float = 0.5,
    save_prob_path: Optional[str] = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    prob = br.predict_mask(image)  # [H,W] in [0,1]
    if save_prob_path:
        Image.fromarray((np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)).save(save_prob_path)
    alpha = (prob >= float(thr)).astype(np.float32)
    rgba = rgba_compose(image, alpha)
    return rgba


def run_batch(
    input_dir: str,
    output_dir: str,
    methods: List[str],  # ["rmbg", "birefnet"]
    birefnet_ckpt: Optional[str] = None,
    thr: float = 0.5,
    limit: Optional[int] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    rmbg = RMBG2() if "rmbg" in methods else None
    br = BiRefNet(ckpt_path=birefnet_ckpt) if "birefnet" in methods else None

    images = list_images(input_dir)
    if limit is not None:
        images = images[: max(0, limit)]

    csv_path = str(Path(output_dir) / "results_baselines.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "method", "params", "time_ms", "saved_path"])  # header

        for img_path in images:
            stem = Path(img_path).stem

            if "rmbg" in methods and rmbg is not None:
                t0 = time.perf_counter()
                try:
                    rgba = run_rmbg(img_path, rmbg)
                    dt_ms = int((time.perf_counter() - t0) * 1000)
                    out_path = str(Path(output_dir) / f"{stem}__rmbg__t{dt_ms}ms.png")
                    rgba.save(out_path)
                    writer.writerow([img_path, "rmbg", "-", dt_ms, out_path])
                    print(f"Saved: {out_path}")
                except Exception as e:
                    dt_ms = int((time.perf_counter() - t0) * 1000)
                    print(f"[ERROR][rmbg] {img_path}: {e}")
                    writer.writerow([img_path, "rmbg", "-", dt_ms, ""]) 

            if "birefnet" in methods and br is not None:
                t0 = time.perf_counter()
                try:
                    prob_path = str(Path(output_dir) / f"{stem}__birefnet_prob.png")
                    rgba = run_birefnet(img_path, br, thr=thr, save_prob_path=prob_path)
                    dt_ms = int((time.perf_counter() - t0) * 1000)
                    out_path = str(Path(output_dir) / f"{stem}__birefnet_thr{thr:.2f}__t{dt_ms}ms.png")
                    rgba.save(out_path)
                    writer.writerow([img_path, "birefnet", f"thr={thr:.2f}", dt_ms, out_path])
                    print(f"Saved: {out_path}")
                except Exception as e:
                    dt_ms = int((time.perf_counter() - t0) * 1000)
                    print(f"[ERROR][birefnet] {img_path}: {e}")
                    writer.writerow([img_path, "birefnet", f"thr={thr:.2f}", dt_ms, ""]) 

    return csv_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch baselines: pure RMBG and pure BiRefNet")
    parser.add_argument("input_dir", type=str, help="输入图片目录（仅一层，不递归）")
    parser.add_argument("output_dir", type=str, help="输出目录")
    parser.add_argument("--methods", type=str, default="rmbg,birefnet", help="逗号分隔：rmbg,birefnet")
    parser.add_argument("--birefnet-ckpt", type=str, default=None, help="BiRefNet 权重(可选)")
    parser.add_argument("--thr", type=float, default=0.5, help="BiRefNet 阈值（生成alpha）")
    parser.add_argument("--limit", type=int, default=None, help="最多处理前 N 张图片")

    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    csv_path = run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        methods=methods,
        birefnet_ckpt=args.birefnet_ckpt,
        thr=args.thr,
        limit=args.limit,
    )
    print(f"CSV saved: {csv_path}")


