import argparse
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from quant_fx.debugger.qxnn_loader import (
    load_cpu_onnx_payloads,
    load_debug_graph_qxnn,
    load_io_contracts,
)

CACHE = os.path.join(tempfile.gettempdir(), "qxnn_cpu")


class Segment:
    def __init__(self, path: str, tag: str):
        self.contracts = load_io_contracts(path)
        self.order = self._topological()
        self.npu = {}
        self.cpu = {}
        payloads = load_cpu_onnx_payloads(path)
        os.makedirs(CACHE, exist_ok=True)

        for name in self.order:
            if self.contracts[name]["device"] == "npu":
                self.npu[name] = load_debug_graph_qxnn(path, name)[0]
                continue

            blob = os.path.join(CACHE, f"{tag}_{name}.onnx")
            if not os.path.exists(blob):
                with open(blob, "wb") as f:
                    f.write(payloads[name])

            self.cpu[name] = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])

    def _topological(self) -> list[str]:
        pending = dict(self.contracts)
        produced = {o["name"] for spec in pending.values() for o in spec["outputs"]}
        ready = {i["name"] for spec in pending.values() for i in spec["inputs"]} - produced
        self.entry = sorted(ready)
        order = []

        while pending:
            for name, spec in list(pending.items()):
                if {i["name"] for i in spec["inputs"]} <= ready:
                    order.append(name)
                    ready |= {o["name"] for o in spec["outputs"]}
                    del pending[name]
                    break
            else:
                raise RuntimeError(f"cannot order partitions: {list(pending)}")

        consumed = {i["name"] for spec in self.contracts.values() for i in spec["inputs"]}
        self.exit = sorted(produced - consumed)

        return order

    # ActQuant covers activations, QConv/QGemm the weights; all ship bypassed, so a run
    # that leaves fp32 on measures float against float
    QUANT = ("ActQuant", "QConv", "QGemm")

    def set_quantized(self, on: bool) -> int:
        found = [m for g in self.npu.values() for _, m in g.named_modules()
                 if type(m).__name__ in self.QUANT]

        for m in found:
            m.fp32 = not on
            if type(m).__name__ == "ActQuant":
                m.observe = False

        return len(found)

    def run(self, tensor: np.ndarray) -> np.ndarray:
        values = {self.entry[0]: tensor}

        for name in self.order:
            spec = self.contracts[name]
            args = [values[i["name"]] for i in spec["inputs"]]

            if name in self.npu:
                with torch.no_grad():
                    out = self.npu[name](*[torch.as_tensor(a) for a in args])
                out = [t.numpy() for t in (out if isinstance(out, (list, tuple)) else [out])]
            else:
                session = self.cpu[name]
                out = session.run(
                    None,
                    {s.name: np.asarray(a, np.float32) for s, a in zip(session.get_inputs(), args)}
                )

            for slot, value in zip(spec["outputs"], out):
                values[slot["name"]] = value

        return values[self.exit[0]]


def preprocess(path: str, size: int) -> np.ndarray:
    rgb = Image.open(path).convert("RGB").resize((size, size))
    bgr = np.asarray(rgb, np.float32)[:, :, ::-1].copy()

    return ((bgr - 127.5) / 127.5).transpose(2, 0, 1)[None]


def agreement(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    a, b = expected.reshape(-1), actual.reshape(-1)
    cosine = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    qsnr = float(20 * np.log10(np.linalg.norm(a) / (np.linalg.norm(a - b) + 1e-12)))

    return cosine, qsnr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="/work/build")
    parser.add_argument("--image", default="/work/calib/dog.jpg")
    parser.add_argument("--size", type=int, default=672)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    segments = [
        Segment(f"{args.build}/segment{i}/quant_diagnosis/segment{i}_ready.qxnn", f"seg{i}")
        for i in range(args.segments)
    ]

    pixel_values = preprocess(args.image, args.size)
    reference = measured = pixel_values

    for index, segment in enumerate(segments):
        segment.set_quantized(False)
        reference = segment.run(reference)
        segment.set_quantized(True)
        measured = segment.run(measured)
        cosine, qsnr = agreement(reference, measured)
        print(f"segment{index} cosine {cosine:.4f} qsnr {qsnr:6.2f} dB", flush=True)

    cosine, qsnr = agreement(reference, measured)
    print(f"encoder  cosine {cosine:.4f} qsnr {qsnr:6.2f} dB", flush=True)

    if args.out:
        np.save(f"{args.out}/hidden_float.npy", reference)
        np.save(f"{args.out}/hidden_int8.npy", measured)


if __name__ == "__main__":
    main()
