from pathlib import Path
import os
from typing import Optional

from PIL import Image

from models.rmbg2 import RMBG2, rgba_compose


def run_zero_shot(image_path: str, output_dir: Optional[str] = "outputs") -> str:
    """
    Zero-training one-click cutout with RMBG-2.0 only.
    - Input: image path
    - Output: RGBA PNG path under output_dir
    """
    os.makedirs(output_dir or "outputs", exist_ok=True)
    out_dir = output_dir or "outputs"

    image = Image.open(image_path).convert("RGB")
    rmbg = RMBG2()
    alpha = rmbg.predict_alpha(image)
    rgba = rgba_compose(image, alpha)
    out_path = str(Path(out_dir) / (Path(image_path).stem + "_rmbg2.png"))
    rgba.save(out_path)
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.zero_shot <image_path> [output_dir]")
        raise SystemExit(1)
    img = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    path = run_zero_shot(img, out)
    print(f"Saved: {path}")


