from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

PATCH, WINDOW, SOFTMAX_LIMIT = 14, 24, 4080

app = typer.Typer(
    help="SAM3 vision encoder compiled for the DEEPX DX-M1 NPU.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
console = Console()


class Attention(str, Enum):
    stock = "stock"
    dxm1 = "dxm1"


Model = Annotated[
    str, typer.Option(envvar="SAM3_MODEL", help="Hugging Face id of the SAM3 checkpoint.")
]
Size = Annotated[int, typer.Option(help="Square input side in pixels.")]
Build = Annotated[
    Path, typer.Option(file_okay=False, help="Folder holding the export and compile outputs.")
]
Picture = Annotated[
    Path, typer.Option("--image", exists=True, dir_okay=False, help="Picture to run.")
]
Prompt = Annotated[str, typer.Option(help="Text prompt for the decoder.")]


def check_size(size: int) -> int:
    grid = size // PATCH

    if size % PATCH or grid % WINDOW or grid * grid > SOFTMAX_LIMIT:
        raise typer.BadParameter(
            f"needs a multiple of {PATCH} whose grid, size / {PATCH}, is a multiple of "
            f"{WINDOW} and squares to at most {SOFTMAX_LIMIT}, such as 672"
        )

    return size


def parse_cuts(cuts: str) -> list[int]:
    try:
        return [int(c) for c in cuts.split(",") if c.strip()]
    except ValueError:
        raise typer.BadParameter("expects layer indexes such as 8,16,24") from None


@app.command(help="Export the encoder as ONNX segments with DX-COM configs and calibration.")
def export(
    model: Model = "facebook/sam3",
    size: Annotated[int, typer.Option(callback=check_size, help="Square input side in pixels.")] = 672,
    cuts: Annotated[
        str, typer.Option(help="Layer indexes that start a new segment.")
    ] = "8,16,24",
    attention: Annotated[
        Attention,
        typer.Option(help="[bold]dxm1[/] works around DX-COM 2.4.1 silicon bugs and is wrong in float."),
    ] = Attention.stock,
    out: Build = Path("build"),
    calib: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Calibration pictures.")
    ] = Path("calib"),
) -> None:
    from sam3_dxnn.export import export as run

    table = Table("Segment", "Layers", "ONNX nodes", title=f"{attention.value} attention")

    with console.status("Exporting segments"):
        for s in run(model, size, parse_cuts(cuts), out, calib, attention is Attention.dxm1):
            console.log(f"segment{s.index}: layers {s.first}-{s.last}, {s.nodes} nodes")
            table.add_row(str(s.index), f"{s.first}-{s.last}", str(s.nodes))

    console.print(table)


@app.command(help="Run the compiled .qxnn graphs in float and INT8 and report the drift.")
def bench(
    build: Build = Path("build"),
    image: Picture = Path("calib/dog.jpg"),
    size: Size = 672,
    save: Annotated[
        bool, typer.Option(help="Write hidden_float.npy and hidden_int8.npy for compare.")
    ] = False,
) -> None:
    from sam3_dxnn.bench import bench as run

    with console.status("Running segments"):
        result = run(build, image, size)

    table = Table("Segment", "Cumulative cosine", "Cumulative QSNR", title="Encoder drift")

    for index, (cosine, qsnr) in enumerate(result.cumulative):
        table.add_row(str(index), f"{cosine:.4f}", f"{qsnr:.2f} dB")

    console.print(table)

    if save:
        np.save(build / "hidden_float.npy", result.reference)
        np.save(build / "hidden_int8.npy", result.measured)


@app.command(help="Decode both encoder outputs and draw the side-by-side picture.")
def compare(
    model: Model = "facebook/sam3",
    build: Build = Path("build"),
    image: Picture = Path("calib/dog.jpg"),
    prompt: Prompt = "dog",
    size: Size = 672,
    threshold: Annotated[float, typer.Option(help="Instance score threshold.")] = 0.4,
    out: Annotated[Path, typer.Option(dir_okay=False, help="Picture to write.")] = Path(
        "assets/compare.png"
    ),
) -> None:
    from sam3_dxnn.compare import compare as run

    with console.status("Decoding masks"):
        runs, iou = run(model, build, image, prompt, size, threshold, out)

    table = Table("", *(r.label for r in runs), title=f"Prompt '{prompt}', {size} px")
    table.add_row("instances", *(str(len(r.masks)) for r in runs))
    table.add_row("mask px", *(f"{r.pixels:,}" for r in runs))
    table.add_row("score", *(", ".join(f"{s:.3f}" for s in r.scores) for r in runs))
    table.add_row("mask IoU", "", "-" if iou is None else f"{iou:.4f}")

    console.print(table)
    console.print(f"Wrote {out}")


@app.command(help="Check the exported float chain against stock SAM3.")
def verify(
    model: Model = "facebook/sam3",
    build: Build = Path("build"),
    image: Picture = Path("calib/dog.jpg"),
    prompt: Prompt = "dog",
    size: Size = 672,
) -> None:
    from sam3_dxnn.verify import verify as run

    with console.status("Running stock SAM3"):
        verdict = run(model, build, image, prompt, size)

    table = Table("", "stock", "exported", title="Against stock SAM3")
    table.add_row("instances", str(len(verdict.stock.masks)), str(len(verdict.exported.masks)))
    table.add_row("mask px", f"{verdict.stock.pixels:,}", f"{verdict.exported.pixels:,}")
    table.add_row("encoder cosine", "", f"{verdict.cosine:.6f}")
    table.add_row("encoder QSNR", "", f"{verdict.qsnr:.2f} dB")
    table.add_row("mask IoU", "", "-" if verdict.iou is None else f"{verdict.iou:.4f}")

    console.print(table)
