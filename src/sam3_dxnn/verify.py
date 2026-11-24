from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3_dxnn.compare import Run
from sam3_dxnn.decode import load_sam3, masks_of
from sam3_dxnn.graph import load_backbone, run_backbone
from sam3_dxnn.metrics import agreement, mask_iou
from sam3_dxnn.pixels import pixel_values


@dataclass
class Verdict:
    cosine: float
    qsnr: float
    stock: Run
    exported: Run
    iou: float | None


def verify(model: str, build: Path, image: Path, prompt: str, size: int) -> Verdict:
    sam3, processor = load_sam3(model)
    native = torch.from_numpy(pixel_values(Image.open(image), size, bgr=False)[None])
    stock = load_backbone(model, size=size, permute=False, bgr=False)

    expected = run_backbone(stock, native)
    chain = np.load(build / "hidden_float.npy")
    cosine, qsnr = agreement(expected, chain)

    runs = [
        Run(label, *masks_of(sam3, processor, hidden, prompt, size, 0.4))
        for label, hidden in (("stock", expected), ("exported", chain))
    ]

    return Verdict(cosine, qsnr, *runs, mask_iou(runs[0].masks, runs[1].masks))
