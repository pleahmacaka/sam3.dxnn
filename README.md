# sam3.dxnn

SAM3's vision encoder compiled for the DeepX DX-M1 NPU.

Weights: [huggingface.co/PleahMaCaka/sam3.dxnn](https://huggingface.co/PleahMaCaka/sam3.dxnn)

## Build

```sh
nix develop
just build
```

Writes four `.dxnn` segments and their quantization checkpoints to `build/`.

## Result

Prompt `dog`, 672 px, decoder in float on the CPU for both paths.

| float | INT8 |
| --- | --- |
| ![float](assets/float.png) | ![int8](assets/int8.png) |
| 1 instance, 4,295,261 px | 1 instance, 4,316,931 px |
| reference | IoU 0.9896 |

`just bench` reports the same comparison per segment.

Encoder QSNR lands at 11-17 dB per segment, which the mask threshold absorbs.
Running the `.dxnn` on hardware is untested: no card.

## Layout

All 32 layers run on the NPU. The only CPU node is the patch reorder in front of
the first segment.

| Segment | Layers | NPU nodes | CPU nodes |
| --- | --- | --- | --- |
| 0 | 0-7 | 559 | 1 |
| 1 | 8-15 | 551 | 0 |
| 2 | 16-23 | 551 | 0 |
| 3 | 24-31 | 552 | 0 |

## Constraints

Input size follows the patch grid, `size / 14`. The grid squared must stay at or
under 4080 or global attention's `Softmax` falls to the CPU, and the grid must be a
multiple of the window size 24 or padding produces a tensor that fails codegen.
Grid 48, so 672 px, satisfies both.

Segments are 8 layers because register-map generation exhausts host memory beyond
that, and each is exported as its own graph taking the previous hidden state, which
is what keeps the earlier layers off the CPU.

SAM3's pairwise rotary embedding is rewritten as the half-split form by permuting
each head's `q_proj` and `k_proj` rows, and `Neg` becomes `Mul(-1)`. Both are exact.
The graph takes BGR in the 0-255 domain to match DX-COM's calibration loader.

Photo by [Alvan Nee](https://unsplash.com/photos/0WJBoHc_EWo) on Unsplash.
