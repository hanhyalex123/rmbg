from pathlib import Path
import os
from typing import Optional

import numpy as np
from PIL import Image

from models.rmbg2 import RMBG2, rgba_compose
from models.birefnet import BiRefNet


def _smooth_mask(image: Image.Image, mask_prob: np.ndarray) -> np.ndarray:
    # 尝试用引导滤波将语义概率对齐边缘，失败则用高斯平滑
    try:
        import cv2
        from fastguidedfilter import fastguidedfilter
        rgb = np.asarray(image).astype(np.float32) / 255.0
        m = fastguidedfilter(rgb, mask_prob.astype(np.float32), r=4, eps=1e-4)
        return np.clip(m, 0.0, 1.0)
    except Exception:
        try:
            import cv2
            return cv2.GaussianBlur(mask_prob.astype(np.float32), (0, 0), 1.2)
        except Exception:
            return mask_prob


def _soft_fuse(alpha: np.ndarray, sem_prob: np.ndarray) -> np.ndarray:
    """仅在 alpha 不确定区域，按语义的“背景置信度”柔和压低 alpha，避免整体变糊。"""
    # alpha 未知带
    unk = (alpha > 0.1) & (alpha < 0.9)
    # 语义“背景置信度” [0,1]：0 表示前景，1 表示强背景
    bg_conf = np.clip((0.5 - sem_prob) / 0.5, 0.0, 1.0)
    # 只在未知带作用
    g = (bg_conf * unk.astype(np.float32))
    # 强度系数，避免过度压低（0~1）
    k = 0.8
    alpha_out = alpha * (1.0 - k * g)
    return np.clip(alpha_out, 0.0, 1.0)


def _hysteresis_gate(alpha: np.ndarray, sem_prob: np.ndarray,
                     t_fg_high: float = 0.85, t_bg_low: float = 0.15,
                     bg_clean_px: int = 3) -> np.ndarray:
    """
    语义“硬门控”：
    - 强前景区(FG anchor)：保持 RMBG alpha（不提亮，避免把毛发填满）。
    - 强背景区(BG anchor)：直接压到 0（保守清理背景同色误抠）。
    - 其它区域：保持 RMBG alpha（必要时再做微精修）。
    """
    import cv2
    F = (sem_prob >= t_fg_high)
    B = (sem_prob <= t_bg_low)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_clean_px, bg_clean_px))
    Bc = cv2.morphologyEx(B.astype(np.uint8), cv2.MORPH_OPEN, k)
    Bc = cv2.morphologyEx(Bc, cv2.MORPH_CLOSE, k)
    out = alpha.copy()
    out[Bc > 0] = 0.0
    return np.clip(out, 0.0, 1.0)


def run_semantic_fuse(
    image_path: str,
    output_dir: Optional[str] = "outputs",
    birefnet_ckpt: Optional[str] = None,
) -> str:
    """
    Zero-training semantic fusion: BiRefNet (semantic FG) + RMBG-2.0 (alpha).
    A' = A * M, then light band refine.
    If no ckpt provided, a generic semantic model is used as fallback.
    """
    os.makedirs(output_dir or "outputs", exist_ok=True)
    out_dir = output_dir or "outputs"

    image = Image.open(image_path).convert("RGB")
    rmbg = RMBG2()
    alpha = rmbg.predict_alpha(image)

    br = BiRefNet(ckpt_path=birefnet_ckpt)
    sem_prob = br.predict_mask(image)  # [H,W] in [0,1]
    sem_prob = _smooth_mask(image, sem_prob)

    # 首先做“硬门控”清理强背景像素，避免前景/背景再度“融合”
    alpha_gate = _hysteresis_gate(alpha, sem_prob, t_fg_high=0.85, t_bg_low=0.15, bg_clean_px=3)
    # 然后可选对未知带做极轻微抑制（降低同色边界的小块残留）。如感觉过度，可注释下一行。
    alpha_fused = _soft_fuse(alpha_gate, sem_prob)
    rgba = rgba_compose(image, alpha_fused)
    out_path = str(Path(out_dir) / (Path(image_path).stem + "_fuse.png"))
    rgba.save(out_path)
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.semantic_fuse <image_path> [output_dir] [birefnet_ckpt]")
        raise SystemExit(1)
    img = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    ck = sys.argv[3] if len(sys.argv) > 3 else None
    p = run_semantic_fuse(img, out, ck)
    print(f"Saved: {p}")


