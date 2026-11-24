import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from quant_fx.debugger.qxnn_loader import (
    load_cpu_onnx_payloads,
    load_debug_graph_qxnn,
    load_io_contracts,
)

from sam3_dxnn.metrics import agreement
from sam3_dxnn.pixels import pixel_values

CACHE = os.path.join(tempfile.gettempdir(), "qxnn_cpu")


@dataclass
class Bench:
    cumulative: list[tuple[float, float]]
    reference: np.ndarray
    measured: np.ndarray


class Segment:

    # all three ship with fp32 on, so a run that leaves any of them on measures float against float
    QUANT = ("ActQuant", "QConv", "QGemm")

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


def bench(build: Path, image: Path, size: int) -> Bench:
    count = len(list(build.glob("segment*.json")))
    segments = [
        Segment(f"{build}/segment{i}/quant_diagnosis/segment{i}_ready.qxnn", f"seg{i}")
        for i in range(count)
    ]

    reference = measured = pixel_values(Image.open(image), size)[None]
    cumulative = []

    for segment in segments:
        segment.set_quantized(False)
        reference = segment.run(reference)
        segment.set_quantized(True)
        measured = segment.run(measured)
        cumulative.append(agreement(reference, measured))

    return Bench(cumulative, reference, measured)
