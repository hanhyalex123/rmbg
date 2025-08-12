import io
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class RMBG2:
    """
    Thin wrapper of briaai/RMBG-2.0 for alpha prediction.
    - Input: PIL.Image RGB
    - Output: numpy float32 alpha in range [0, 1], size equals to original image
    """

    def __init__(self, device: str = "cuda", input_size: Tuple[int, int] = (1024, 1024)) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True
        )
        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
        self.model.to(self.device)
        self.model.eval()
        self._transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        preds = self.model(tensor)[-1].sigmoid().detach().cpu()
        alpha_small = preds[0].squeeze().numpy().astype(np.float32)  # HxW in [0,1]
        # resize back to original size
        alpha_img = Image.fromarray((alpha_small * 255).astype(np.uint8))
        alpha_full = alpha_img.resize(rgb.size, resample=Image.BILINEAR)
        alpha = np.asarray(alpha_full).astype(np.float32) / 255.0
        return np.clip(alpha, 0.0, 1.0)


def rgba_compose(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Compose an RGBA image using the given alpha (H, W) in [0,1]."""
    rgb = image.convert("RGB")
    a = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    a_img = Image.fromarray(a).resize(rgb.size, resample=Image.BILINEAR)
    r, g, b = rgb.split()
    rgba = Image.merge("RGBA", (r, g, b, a_img))
    return rgba


