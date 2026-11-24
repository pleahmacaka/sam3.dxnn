import torch
import transformers.models.sam3.modeling_sam3 as sam3
from transformers import AttentionInterface, Sam3Model

HEADS, HEAD_DIM = 16, 64
PAIRWISE_ROPE = sam3.apply_rotary_pos_emb_2d
PERM = torch.tensor([*range(0, HEAD_DIM, 2), *range(1, HEAD_DIM, 2)])

# probabilities below CLIP_ROWS / row length survive int8 only on a scale of their own
CLIP_ROWS = 8.0


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


def check_rope_equivalence() -> tuple[float, float]:
    q = torch.randn(1, HEADS, 8, HEAD_DIM)
    k = torch.randn(1, HEADS, 8, HEAD_DIM)
    cos, sin = torch.randn(8, HEAD_DIM), torch.randn(8, HEAD_DIM)
    qa, ka = PAIRWISE_ROPE(q, k, cos, sin)
    qb, kb = apply_rope_half(q.index_select(-1, PERM), k.index_select(-1, PERM), cos, sin)
    gap_q = (qa.index_select(-1, PERM) - qb).abs().max().item()
    gap_k = (ka.index_select(-1, PERM) - kb).abs().max().item()
    assert gap_q < 1e-5 and gap_k < 1e-5, "half-split rope is not equivalent to pairwise rope"

    return gap_q, gap_k


def use_half_rope():
    sam3.apply_rotary_pos_emb_2d = apply_rope_half


def dxm1_attention(module, query, key, value, attention_mask=None, scaling=None, **kwargs):
    windows, heads, length, width = value.shape
    probs = torch.softmax((query @ key.transpose(-1, -2)) * scaling, dim=-1)

    # DX-COM 2.4.1 silicon reads V's (window, head) batch dims swapped in windowed attention
    if windows > 1:
        value = value.reshape(heads, windows, length, width).transpose(0, 1)
        value = value.reshape(windows, heads, length, width)

    high = torch.relu(probs - CLIP_ROWS / length)
    # two MatMuls sharing one V fail DX-COM's OPTIMIZE phase, so the second gets its own copy
    out = (probs - high) @ value + (high @ (value * 2.0)) * 0.5

    return out.transpose(1, 2).contiguous(), None


def register_dxm1_attention():
    AttentionInterface.register("dxm1", dxm1_attention)


def permute_proj(linear):
    w = linear.weight.data.view(HEADS, HEAD_DIM, -1)
    linear.weight.data = w.index_select(1, PERM).reshape(linear.weight.shape)

    if linear.bias is not None:
        b = linear.bias.data.view(HEADS, HEAD_DIM)
        linear.bias.data = b.index_select(1, PERM).reshape(linear.bias.shape)


def accept_bgr(backbone):
    # DX-COM's calibration loader reads images with cv2, so the compiled graph is fed BGR
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


def run_backbone(backbone, pixel_values):
    patch = backbone.config.patch_size
    rows, columns = pixel_values.shape[-2] // patch, pixel_values.shape[-1] // patch

    with torch.no_grad():
        hidden = backbone.embeddings(pixel_values).view(1, rows, columns, -1)
        hidden = backbone.layer_norm(hidden)

        for layer in backbone.layers:
            hidden = layer(hidden)

    return hidden.reshape(1, rows * columns, -1).numpy()


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


def demo():
    q, k, v = (torch.randn(1, HEADS, 32, HEAD_DIM) for _ in range(3))
    out, _ = dxm1_attention(None, q, k, v, scaling=HEAD_DIM**-0.5)
    expected = torch.softmax(q @ k.transpose(-1, -2) * HEAD_DIM**-0.5, dim=-1) @ v
    assert torch.allclose(out, expected.transpose(1, 2), atol=1e-5)


if __name__ == "__main__":
    demo()
