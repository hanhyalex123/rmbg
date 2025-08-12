from pathlib import Path
import os
from typing import Optional

import numpy as np
from PIL import Image

from models.rmbg2 import RMBG2, rgba_compose
from models.refiner import TinyRefiner


def _auto_trimap(alpha: np.ndarray, low: float = 0.05, high: float = 0.95, band_px: int = 8) -> np.ndarray:
    import cv2
    fg = (alpha >= high).astype(np.uint8)
    bg = (alpha <= low).astype(np.uint8)
    unk = 1 - np.clip(fg + bg, 0, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px, band_px))
    unk = cv2.dilate(unk, kernel) * (1 - fg) * (1 - bg)
    trimap = fg * 255 + unk * 128
    return trimap.astype(np.uint8)


def _guided_refine(rgb: np.ndarray, alpha: np.ndarray, radius: int = 4, eps: float = 1e-4) -> np.ndarray:
    # 依赖可选，如果不可用直接回退
    try:
        from fastguidedfilter import fastguidedfilter
        a = fastguidedfilter(rgb.astype(np.float32) / 255.0, alpha.astype(np.float32), radius, eps)
        return np.clip(a, 0.0, 1.0)
    except Exception:
        return alpha


def run_micro_refine(image_path: str, output_dir: Optional[str] = "outputs", ckpt_path: Optional[str] = None) -> str:
    """
    Micro-refine version (no training):
    - RMBG-2.0 for alpha
    - Guided filter refinement on unknown band
    """
    os.makedirs(output_dir or "outputs", exist_ok=True)
    out_dir = output_dir or "outputs"

    image = Image.open(image_path).convert("RGB")
    rgb_np = np.asarray(image)
    rmbg = RMBG2()
    alpha = rmbg.predict_alpha(image)

    # build trimap and refine alpha only in unknown band
    trimap = _auto_trimap(alpha)
    unk = (trimap == 128)
    refined = _guided_refine(rgb_np, alpha, radius=4, eps=1e-4)
    alpha_mid = np.where(unk, refined, alpha)

    # optional: load tiny refiner for residual correction on unknown band
    if ckpt_path and os.path.exists(ckpt_path):
        import torch
        model = TinyRefiner(in_ch=4).eval().to("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(ckpt_path, map_location=model.device if hasattr(model, 'device') else 'cpu')
        model.load_state_dict(state, strict=False)
        # prepare tensor
        x_img = (rgb_np.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        x_alpha = alpha_mid.astype(np.float32)[None, None, ...]
        x = np.concatenate([x_img, x_alpha], axis=1)
        xt = torch.from_numpy(x)
        if next(model.parameters()).is_cuda:
            xt = xt.cuda()
        with torch.no_grad():
            delta = model(xt).cpu().numpy()[0, 0]
        alpha_out = np.clip(alpha_mid + delta, 0.0, 1.0)
    else:
        alpha_out = alpha_mid

    rgba = rgba_compose(image, alpha_out)
    out_path = str(Path(out_dir) / (Path(image_path).stem + "_rmbg2_refined.png"))
    rgba.save(out_path)
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.micro_refine <image_path> [output_dir] [ckpt_path]")
        raise SystemExit(1)
    img = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    ck = sys.argv[3] if len(sys.argv) > 3 else None
    path = run_micro_refine(img, out, ck)
    print(f"Saved: {path}")


