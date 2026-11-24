from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_dxnn.decode import load_sam3, masks_of
from sam3_dxnn.metrics import mask_iou
from sam3_dxnn.render import compose, fit, overlay

PATHS = (
    ("float", "hidden_float", "ONNX (CPU, float)", (70, 200, 255)),
    ("INT8", "hidden_int8", "DXNN (INT8, DX-COM scales)", (255, 150, 60)),
)


@dataclass
class Run:
    label: str
    masks: list[np.ndarray]
    scores: list[float]

    @property
    def pixels(self) -> int:
        return int(sum(m.sum() for m in self.masks))


def compare(
    model: str, build: Path, image: Path, prompt: str, size: int, threshold: float, out: Path
) -> tuple[list[Run], float | None]:
    sam3, processor = load_sam3(model)
    picture = fit(Image.open(image), size)
    runs, panels, captions = [], [], []

    for label, stem, heading, colour in PATHS:
        hidden = np.load(build / f"{stem}.npy")
        run = Run(label, *masks_of(sam3, processor, hidden, prompt, size, threshold))
        noun = "instance" if len(run.masks) == 1 else "instances"

        runs.append(run)
        panels.append(overlay(picture, run.masks, colour))
        captions.append((heading, f"{len(run.masks)} {noun}, {run.pixels:,} px"))

    out.parent.mkdir(parents=True, exist_ok=True)
    compose(panels, captions).save(out)

    return runs, mask_iou(runs[0].masks, runs[1].masks)
