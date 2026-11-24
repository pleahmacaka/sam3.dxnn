import glob
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper, shape_inference
from onnxsim import simplify
from PIL import Image

from sam3_dxnn.graph import (
    Head,
    Tail,
    check_rope_equivalence,
    load_backbone,
    register_dxm1_attention,
    use_half_rope,
)
from sam3_dxnn.pixels import MEAN, pixel_values


@dataclass
class Segment:
    index: int
    first: int
    last: int
    nodes: int


def crops_of(image):
    width, height = image.size
    half_w, half_h = width // 2, height // 2

    return [image,
            image.crop((0, 0, half_w, half_h)),
            image.crop((half_w, 0, width, half_h)),
            image.crop((0, half_h, half_w, height)),
            image.crop((half_w, half_h, width, height))]


def load_calibration(folder: Path, size: int) -> torch.Tensor:
    files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    frames = [pixel_values(view, size) for name in files for view in crops_of(Image.open(name))]

    return torch.from_numpy(np.stack(frames))


def image_config(size, samples):
    return {
        "inputs": {"pixel_values": [1, 3, size, size]},
        "calibration_method": "minmax",
        "calibration_num": samples,
        "default_loader": {
            "dataset_path": "/work/calib",
            "file_extensions": ["png", "jpg"],
            "preprocessings": [
                {"resize": {"width": size, "height": size}},
                {"normalize": {"mean": [MEAN] * 3, "std": [MEAN] * 3}},
                {"transpose": {"axis": [2, 0, 1]}},
            ],
        },
    }


def hidden_config(grid, width, samples):
    return {
        "inputs": {"hidden": [1, grid, grid, width]},
        "calibration_method": "minmax",
        "calibration_num": samples,
        "custom_loader": {"package": "calibration.Tensors"},
    }


def simplify_and_save(module, example, stem, name):
    torch.onnx.export(module, example, f"{stem}.onnx", opset_version=17, dynamo=False,
                      input_names=[name], output_names=["output"])
    model, _ = simplify(onnx.load(f"{stem}.onnx"),
                        overwrite_input_shapes={name: list(example.shape)})
    model.graph.initializer.append(numpy_helper.from_array(np.array(-1.0, np.float32), "neg_const"))

    for node in model.graph.node:
        if node.op_type == "Neg":
            node.op_type = "Mul"
            node.input.append("neg_const")

    # onnxsim leaves rank-0 value_info on fused Gemm outputs, and inference keeps what it finds
    del model.graph.value_info[:]
    model = shape_inference.infer_shapes(model, strict_mode=False)
    base = os.path.basename(stem)
    onnx.save(model, f"{stem}_ready.onnx", save_as_external_data=True,
              all_tensors_to_one_file=True, location=f"{base}_ready.data")

    return len(model.graph.node)


def export(
    model: str, size: int, cuts: list[int], out: Path, calib: Path, dxm1: bool
) -> Iterator[Segment]:
    check_rope_equivalence()
    use_half_rope()
    register_dxm1_attention()

    backbone = load_backbone(model, size=size)
    stock = backbone.config._attn_implementation
    traced = "dxm1" if dxm1 else stock
    bounds = [0, *cuts, len(backbone.layers)]
    grid, width = size // backbone.config.patch_size, backbone.config.hidden_size

    out.mkdir(parents=True, exist_ok=True)
    batch = load_calibration(calib, size)

    for index, (start, end) in enumerate(pairwise(bounds)):
        stem = os.path.join(out, f"segment{index}")
        backbone.config._attn_implementation = traced

        if index == 0:
            module = Head(backbone, end)
            nodes = simplify_and_save(module, batch[:1], stem, "pixel_values")
            config = image_config(size, len(batch))
        else:
            module = Tail(backbone, start, end)
            nodes = simplify_and_save(module, batch[:1], stem, "hidden")
            np.save(f"{stem}_calib.npy", batch.numpy())
            config = hidden_config(grid, width, len(batch))

        # the dxm1 attention is wrong in float on purpose, so calibration flows through the stock one
        backbone.config._attn_implementation = stock

        with torch.no_grad():
            batch = torch.cat([module(batch[i:i + 1]) for i in range(len(batch))])

        with open(f"{stem}.json", "w") as f:
            json.dump(config, f, indent=2)

        yield Segment(index, start, end - 1, nodes)
