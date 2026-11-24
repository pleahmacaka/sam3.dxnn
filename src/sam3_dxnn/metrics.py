import numpy as np


def agreement(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    a, b = expected.reshape(-1).astype(np.float64), actual.reshape(-1).astype(np.float64)
    cosine = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    qsnr = float(20 * np.log10(np.linalg.norm(a) / (np.linalg.norm(a - b) + 1e-12)))

    return cosine, qsnr


def mask_iou(first: list[np.ndarray], second: list[np.ndarray]) -> float | None:
    if not first or not second:
        return None

    a, b = np.logical_or.reduce(first), np.logical_or.reduce(second)

    return float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())
