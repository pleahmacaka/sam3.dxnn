# sam3.dxnn

SAM3's vision encoder compiled for the DeepX DX-M1 NPU.

Weights: [huggingface.co/PleahMaCaka/sam3.dxnn](https://huggingface.co/PleahMaCaka/sam3.dxnn)

## Build

```sh
nix develop
just build
```

Writes four `.dxnn` segments and their quantization checkpoints to `build/`.
`export` runs on the host through uv; `compile` and `bench` run inside the DX-COM
image built from `dxcom/Containerfile`.

## Commands

Every step is a subcommand of the `sam3-dxnn` CLI, and each `just` recipe wraps one.

| Recipe | Command | Runs |
| --- | --- | --- |
| `just export` | `sam3-dxnn export` | ONNX segments, DX-COM configs, calibration tensors |
| `just compile` | `dxcom` | one `.dxnn` and `.qxnn` per segment |
| `just bench` | `sam3-dxnn bench --save` | `.qxnn` in float and INT8, cumulative drift |
| `just compare` | `sam3-dxnn compare` | both encoder outputs through the decoder, `assets/compare.png` |
| `just verify` | `sam3-dxnn verify` | exported float chain against stock SAM3 |

`uv run sam3-dxnn <command> --help` lists the options. `SAM3_MODEL`, `SAM3_SIZE`,
`SAM3_CUTS` and `SAM3_ATTENTION` override the recipe defaults.

```
src/sam3_dxnn/
  cli.py       Typer entry point
  graph.py     backbone rewrites: half-split RoPE, BGR input, DX-M1 attention
  export.py    segment export and DX-COM configs
  bench.py     .qxnn runner on DX-COM's simulator
  decode.py    SAM3 neck and decoder on a given encoder output
  compare.py   side-by-side picture
  verify.py    float chain against stock SAM3
  render.py    picture drawing
  pixels.py    input normalisation
  metrics.py   cosine, QSNR, mask IoU
dxcom/
  Containerfile   DX-COM image for compile and bench
  calibration.py  hidden-state loader DX-COM imports as calibration.Tensors
```

## Result

Prompt `dog`, 672 px, decoder in float on the CPU for both paths.

![compare](assets/compare.png)

| | float | INT8 |
| --- | --- | --- |
| instances | 1 | 1 |
| mask px | 95,199 | 95,413 |
| score | 0.962 | 0.967 |
| mask IoU | | 0.9938 |

The INT8 side runs the `.qxnn` graphs with `fp32` off on `ActQuant`, `QConv` and
`QGemm`, so activations and weights both round on DX-COM's own calibrated
scales. All three ship bypassed, and a run that leaves any of them on measures
float against float.

`just verify` checks the float side against stock SAM3 at the same size: cosine
0.999969, mask IoU 0.9999. The RoPE rewrite and the BGR swap cost nothing.

`just bench` prints the encoder error as it accumulates. Each segment is fine on
its own; the error compounds down the chain.

| Segment | isolated QSNR | cumulative QSNR | cumulative cosine |
| --- | --- | --- | --- |
| 0 | 13.61 dB | 11.50 dB | 0.9673 |
| 1 | 13.90 dB | 7.27 dB | 0.9074 |
| 2 | 18.83 dB | 6.28 dB | 0.8780 |
| 3 | 14.94 dB | 5.14 dB | 0.8472 |

Isolated is `quant_diagnosis/diagnosis_report.html`, which feeds each segment a
float input. Cumulative chains the quantized output forward, which is what the
device would do. The mask survives it: the decoder absorbs an encoder that has
drifted to cosine 0.847.

## Hardware

Simulation does not predict the card. Builds of this graph with DX-COM 2.4.1 ran on a
DX-M1 and drifted far more than the `.qxnn` numbers above:

- Windowed attention reads V with its (window, head) batch dims swapped, a
  miscompile the simulator does not reproduce. The four-segment chain reached
  cosine 0.154 against float.
- INT8 rounds most softmax probabilities to zero: a row of 576 averages 0.0017.

`--attention dxm1` exports a graph that pre-permutes V and splits each probability
row at 8 / row length into two MatMuls with their own scales. That pair measured
chain cosine 0.976 on the card. The split is exact in float, but the permutation is
wrong in float on purpose, so `bench`, `compare` and `verify` only describe the
default `stock` build. A `dxm1` build from this tree has not run on a card, and
`dxcom/Containerfile` still pins DX-COM 2.4.0, the version the numbers above come from.

## NPU layout

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
The graph takes BGR normalised to [-1, 1], the order DX-COM's cv2-based calibration
loader reads.

Photo by [Alvan Nee](https://unsplash.com/photos/0WJBoHc_EWo) on Unsplash.
