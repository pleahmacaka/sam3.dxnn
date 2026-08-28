import argparse
import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from quant_fx.debugger.qxnn_loader import list_diagnosable_graphs_qxnn, load_debug_graph_qxnn

parser = argparse.ArgumentParser()
parser.add_argument("--build", default="/work/build")
parser.add_argument("--segments", type=int, default=4)
parser.add_argument("--image", default="/work/calib/dog.jpg")
parser.add_argument("--size", type=int, default=672)
args = parser.parse_args()


def preprocess(path, size):
    bgr = np.asarray(Image.open(path).convert("RGB").resize((size, size)), np.float32)[:, :, ::-1]

    return ((bgr.copy() - 127.5) / 127.5).transpose(2, 0, 1)[None]


def scales_of(index):
    checkpoint = glob.glob(f"{args.build}/segment{index}/quant_diagnosis/*.qxnn")[0]
    graph = list_diagnosable_graphs_qxnn(checkpoint)[0]
    module, _ = load_debug_graph_qxnn(checkpoint, graph)

    return [float(m.scale) for _, m in module.named_modules() if type(m).__name__ == "ActQuant"]


def quantize(tensor, scale):
    return np.clip(np.round(tensor / scale), -128, 127) * scale


def run(index, tensor, scales=None):
    session = ort.InferenceSession(f"{args.build}/segment{index}_ready.onnx",
                                   providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]

    return output if scales is None else quantize(output, max(scales))


pixel_values = preprocess(args.image, args.size)
reference, measured = pixel_values, pixel_values

for index in range(args.segments):
    reference = run(index, reference)
    measured = run(index, measured, scales_of(index))
    a, b = reference.reshape(-1), measured.reshape(-1)
    cosine = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    qsnr = float(20 * np.log10(np.linalg.norm(a) / (np.linalg.norm(a - b) + 1e-12)))
    print(f"segment{index} cosine {cosine:.4f} qsnr {qsnr:6.2f} dB", flush=True)

a, b = reference.reshape(-1), measured.reshape(-1)
cosine = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
qsnr = float(20 * np.log10(np.linalg.norm(a) / (np.linalg.norm(a - b) + 1e-12)))
print(f"encoder  cosine {cosine:.4f} qsnr {qsnr:6.2f} dB", flush=True)
