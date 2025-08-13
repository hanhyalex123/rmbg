from pathlib import Path
import os
from typing import Optional

import numpy as np
from PIL import Image

from models.birefnet import BiRefNet
from models.rmbg2 import rgba_compose


def run_birefnet_only(
    image_path: str,
    output_dir: Optional[str] = "outputs",
    birefnet_ckpt: Optional[str] = None,
    binarize: bool = False,
    thr: float = 0.5,
) -> str:
    """
    纯 BiRefNet 前景提取：输出 RGBA（alpha 来源于二值/概率前景）。
    - binarize=False: 直接用概率作为 alpha（平滑过渡，边界更柔和）
    - binarize=True: 以阈值 thr 产生二值 alpha（边界更干净，半透明损失）
    """
    os.makedirs(output_dir or "outputs", exist_ok=True)
    out_dir = output_dir or "outputs"

    image = Image.open(image_path).convert("RGB")

    br = BiRefNet(ckpt_path=birefnet_ckpt)
    mask_prob = br.predict_mask(image)  # [H,W] in [0,1]

    if binarize:
        alpha = (mask_prob >= float(thr)).astype(np.float32)
    else:
        alpha = mask_prob.astype(np.float32)

    rgba = rgba_compose(image, alpha)
    suffix = "_birefnet_bin" if binarize else "_birefnet_prob"
    out_path = str(Path(out_dir) / (Path(image_path).stem + f"{suffix}.png"))
    rgba.save(out_path)
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.birefnet_only <image_path> [output_dir] [birefnet_ckpt] [binarize:0|1] [thr]")
        raise SystemExit(1)
    img = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    ck = sys.argv[3] if len(sys.argv) > 3 else None
    bz = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    th = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    path = run_birefnet_only(img, out, ck, bz, th)
    print(f"Saved: {path}")


