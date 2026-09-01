import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from compare import masks_of, vision_output
from export import load_backbone

SIZE, PROMPT = 672, "dog"


def reference_hidden(backbone, pixel_values):
    patch = backbone.config.patch_size
    rows, columns = pixel_values.shape[-2] // patch, pixel_values.shape[-1] // patch

    with torch.no_grad():
        hidden = backbone.embeddings(pixel_values).view(1, rows, columns, -1)
        hidden = backbone.layer_norm(hidden)

        for layer in backbone.layers:
            hidden = layer(hidden)

    return hidden.reshape(1, rows * columns, -1).numpy()


model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float32).eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")

picture = Image.open("calib/dog.jpg").convert("RGB").resize((SIZE, SIZE))
shape = picture.size[::-1]
rgb = np.asarray(picture, np.float32)
native = torch.from_numpy(((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None].copy())

stock = load_backbone("facebook/sam3", size=SIZE, permute=False, bgr=False)
expected = reference_hidden(stock, native)
chain = np.load("build/hidden_float.npy")

a, b = expected.reshape(-1), chain.reshape(-1)
cosine = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
qsnr = float(20 * np.log10(np.linalg.norm(a) / (np.linalg.norm(a - b) + 1e-12)))
print(f"exported float vs stock backbone: cosine {cosine:.6f}, qsnr {qsnr:.2f} dB")

pairs = {
    "stock": masks_of(model, processor, vision_output(model, expected, SIZE), PROMPT, shape, 0.4),
    "exported": masks_of(model, processor, vision_output(model, chain, SIZE), PROMPT, shape, 0.4),
}

for label, (masks, scores) in pairs.items():
    print(f"{label:9} {len(masks)} instances, {int(sum(m.sum() for m in masks))} px, {scores}")

if pairs["stock"][0] and pairs["exported"][0]:
    x = np.logical_or.reduce(pairs["stock"][0])
    y = np.logical_or.reduce(pairs["exported"][0])
    print(f"mask IoU {np.logical_and(x, y).sum() / np.logical_or(x, y).sum():.4f}")
