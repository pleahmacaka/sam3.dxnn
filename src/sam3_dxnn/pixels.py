import numpy as np
from PIL import Image

MEAN = 127.5


def pixel_values(image: Image.Image, size: int, bgr: bool = True) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB").resize((size, size)), np.float32)
    channels = rgb[:, :, ::-1] if bgr else rgb

    return np.ascontiguousarray(((channels - MEAN) / MEAN).transpose(2, 0, 1))
