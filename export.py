import argparse
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import onnx
import torch
import transformers.models.sam3.modeling_sam3 as sam3
from onnx import numpy_helper, shape_inference
from onnxsim import simplify
from PIL import Image
from transformers import Sam3Model

HEADS, HEAD_DIM = 16, 64

PAIRWISE_ROPE = sam3.apply_rotary_pos_emb_2d
PERM = torch.tensor([*range(0, HEAD_DIM, 2), *range(1, HEAD_DIM, 2)])


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)

    return torch.cat((-b, a), dim=-1)


def apply_rope_half(q, k, cos, sin):
    cos = cos.index_select(-1, PERM.to(cos.device))
    sin = sin.index_select(-1, PERM.to(sin.device))
    qf, kf = q.float(), k.float()
    qe = qf * cos + rotate_half(qf) * sin
    ke = kf * cos + rotate_half(kf) * sin

    return qe.type_as(q), ke.type_as(k)


def permute_proj(linear):
    w = linear.weight.data.view(HEADS, HEAD_DIM, -1)
    linear.weight.data = w.index_select(1, PERM).reshape(linear.weight.shape)

    if linear.bias is not None:
        b = linear.bias.data.view(HEADS, HEAD_DIM)
        linear.bias.data = b.index_select(1, PERM).reshape(linear.bias.shape)


def check_rope_equivalence():
    q = torch.randn(1, HEADS, 8, HEAD_DIM)
    k = torch.randn(1, HEADS, 8, HEAD_DIM)
    cos, sin = torch.randn(8, HEAD_DIM), torch.randn(8, HEAD_DIM)
    qa, ka = PAIRWISE_ROPE(q, k, cos, sin)
    qb, kb = apply_rope_half(q.index_select(-1, PERM), k.index_select(-1, PERM), cos, sin)
    gap_q = (qa.index_select(-1, PERM) - qb).abs().max().item()
    gap_k = (ka.index_select(-1, PERM) - kb).abs().max().item()
    print(f"rope equivalence max gap q={gap_q:.2e} k={gap_k:.2e}", flush=True)
    assert gap_q < 1e-5 and gap_k < 1e-5, "half-split rope is not equivalent to pairwise rope"


def accept_bgr(backbone):
    """DX-COM's loader reads images with cv2, so the compiled graph is fed BGR."""
    conv = backbone.embeddings.patch_embeddings.projection
    conv.weight.data = conv.weight.data[:, [2, 1, 0]].contiguous()


def resize_rope(backbone, size):
    config = backbone.config
    grid = size // config.patch_size

    for layer in backbone.layers:
        if layer.window_size == 0:
            layer.rotary_emb = sam3.Sam3ViTRotaryEmbedding(
                config, end_x=grid, end_y=grid, scale=config.window_size / grid)


def load_backbone(model, size=1008, permute=True, bgr=True):
    backbone = Sam3Model.from_pretrained(model, dtype=torch.float32).vision_encoder.backbone

    if size != backbone.config.image_size:
        resize_rope(backbone, size)

    if permute:
        for layer in backbone.layers:
            permute_proj(layer.attention.q_proj)
            permute_proj(layer.attention.k_proj)

    if bgr:
        accept_bgr(backbone)

    backbone.eval()

    return backbone


class Head(torch.nn.Module):

    def __init__(self, backbone, end):
        super().__init__()
        self.embeddings = backbone.embeddings
        self.layer_norm = backbone.layer_norm
        self.layers = torch.nn.ModuleList(list(backbone.layers)[:end])
        self.patch = backbone.config.patch_size

    def forward(self, pixel_values):
        rows = pixel_values.shape[-2] // self.patch
        columns = pixel_values.shape[-1] // self.patch
        hidden = self.embeddings(pixel_values).view(1, rows, columns, -1)
        hidden = self.layer_norm(hidden)

        for layer in self.layers:
            hidden = layer(hidden)

        return hidden


class Tail(torch.nn.Module):

    def __init__(self, backbone, start, end):
        super().__init__()
        self.layers = torch.nn.ModuleList(list(backbone.layers)[start:end])
        self.last = end == len(backbone.layers)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)

        if self.last:
            batch, rows, columns, width = hidden.shape
            return hidden.reshape(batch, rows * columns, width)

        return hidden


def crops_of(image):
    width, height = image.size
    half_w, half_h = width // 2, height // 2

    return [image,
            image.crop((0, 0, half_w, half_h)),
            image.crop((half_w, 0, width, half_h)),
            image.crop((0, half_h, half_w, height)),
            image.crop((half_w, half_h, width, height))]


def load_calibration(folder, size):
    mean = np.array([127.5, 127.5, 127.5], np.float32)
    files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    frames = []

    for name in files:
        for view in crops_of(Image.open(name).convert("RGB")):
            bgr = np.asarray(view.resize((size, size)), np.float32)[:, :, ::-1]
            frames.append(((bgr.copy() - mean) / mean).transpose(2, 0, 1))

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
                {"normalize": {"mean": [127.5, 127.5, 127.5], "std": [127.5, 127.5, 127.5]}},
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

    model = shape_inference.infer_shapes(model, strict_mode=False)
    base = os.path.basename(stem)
    onnx.save(model, f"{stem}_ready.onnx", save_as_external_data=True,
              all_tensors_to_one_file=True, location=f"{base}_ready.data")

    return len(model.graph.node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("SAM3_MODEL", "facebook/sam3"))
    parser.add_argument("--size", type=int, default=882)
    parser.add_argument("--cuts", default="12,24", help="layer indexes that start a new segment")
    parser.add_argument("--out", default="build")
    parser.add_argument("--calib", default="calib")
    args = parser.parse_args()

    check_rope_equivalence()
    sam3.apply_rotary_pos_emb_2d = apply_rope_half

    backbone = load_backbone(args.model, size=args.size)
    total = len(backbone.layers)
    bounds = [0, *(int(c) for c in args.cuts.split(",") if c), total]
    grid, width = args.size // backbone.config.patch_size, backbone.config.hidden_size

    os.makedirs(args.out, exist_ok=True)
    batch = load_calibration(args.calib, args.size)
    print(f"calibration images: {len(batch)}", flush=True)

    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        stem = os.path.join(args.out, f"segment{index}")

        if index == 0:
            module = Head(backbone, end)
            nodes = simplify_and_save(module, batch[:1], stem, "pixel_values")
            config = image_config(args.size, len(batch))
        else:
            module = Tail(backbone, start, end)
            nodes = simplify_and_save(module, batch[:1], stem, "hidden")
            path = os.path.join(args.out, f"segment{index}_calib.npy")
            np.save(path, batch.numpy())
            config = hidden_config(grid, width, len(batch))

        with torch.no_grad():
            batch = torch.cat([module(batch[i:i + 1]) for i in range(len(batch))])

        with open(f"{stem}.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"segment{index}: layers {start}-{end - 1}, {nodes} nodes", flush=True)


if __name__ == "__main__":
    main()
