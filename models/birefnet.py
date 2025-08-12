from typing import Tuple, Optional

import numpy as np
from PIL import Image
import torch


class BiRefNet:
    """
    Minimal BiRefNet inference wrapper.
    NOTE: This expects you to provide a torch .pth checkpoint and a model builder.
    For quick start we implement a very light placeholder using timm backbones if needed.
    You can later replace `self._build_model()` with the official implementation.
    Repo: https://github.com/ZhengPeng7/BiRefNet
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        input_size: Tuple[int, int] = (1024, 1024),
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.model = self._build_model()
        self.model.to(self.device).eval()
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=self.device)
            # allow both plain dict and {"state_dict": ...}
            state_dict = state.get("state_dict", state)
            self.model.load_state_dict(state_dict, strict=False)

    def _build_model(self) -> torch.nn.Module:
        # Placeholder tiny semantic foreground segmenter
        # Users should replace with the official BiRefNet architecture for best results
        return torch.hub.load(
            "pytorch/vision", "fcn_resnet50", pretrained=True
        )

    @torch.inference_mode()
    def predict_mask(self, image: Image.Image) -> np.ndarray:
        from torchvision import transforms

        rgb = image.convert("RGB")
        tfm = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        tensor = tfm(rgb).unsqueeze(0).to(self.device)
        out = self.model(tensor)["out"].softmax(dim=1).detach().cpu()  # [1,C,h,w]
        # naive foreground = sum of non-background; background assumed index 0
        prob = 1.0 - out[0, 0].numpy()
        prob_img = Image.fromarray((prob * 255).astype(np.uint8))
        prob_full = prob_img.resize(rgb.size, resample=Image.BILINEAR)
        mask = np.asarray(prob_full).astype(np.float32) / 255.0
        return np.clip(mask, 0.0, 1.0)


