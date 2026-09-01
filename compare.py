import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

FONTS = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")
MASK_COLOURS = ((70, 200, 255), (255, 150, 60))
PAD, BAND, GAP = 18, 62, 14


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join(FONTS, f"Pretendard-{weight}.otf"), size)


def vision_output(model: Sam3Model, hidden: np.ndarray, size: int) -> Sam3VisionEncoderOutput:
    states = torch.from_numpy(hidden).float()
    grid = size // model.config.vision_config.backbone_config.patch_size
    spatial = states.view(1, grid, grid, -1).permute(0, 3, 1, 2)
    fpn, position = model.vision_encoder.neck(spatial)

    return Sam3VisionEncoderOutput(
        last_hidden_state=states,
        fpn_hidden_states=fpn,
        fpn_position_encoding=position,
    )


def masks_of(model, processor, vision, prompt, size, threshold):
    text = processor(text=prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            vision_embeds=vision,
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"],
        )

    found = processor.post_process_instance_segmentation(outputs, threshold, 0.5, [size])[0]

    return [m.cpu().numpy().astype(bool) for m in found["masks"]], [float(s) for s in found["scores"]]


def overlay(image: Image.Image, masks, colour) -> Image.Image:
    canvas = np.asarray(image.convert("RGB"), np.float32)

    for mask in masks:
        canvas[mask] = canvas[mask] * 0.45 + np.array(colour, np.float32) * 0.55

    out = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(out)

    for mask in masks:
        rows, columns = np.where(mask)
        if len(rows):
            draw.rectangle(
                [columns.min(), rows.min(), columns.max(), rows.max()], outline=colour, width=3
            )

    return out


def centred(draw, box, text, typeface, fill):
    left, top, right, bottom = box
    width = draw.textlength(text, font=typeface)
    ascent, descent = typeface.getmetrics()
    x = left + (right - left - width) / 2
    y = top + (bottom - top - (ascent + descent)) / 2

    draw.text((x, y), text, font=typeface, fill=fill)


def compose(panels, captions) -> Image.Image:
    width, height = panels[0].size
    total_w = PAD * 2 + width * 2 + GAP
    total_h = PAD * 2 + BAND + height
    sheet = Image.new("RGB", (total_w, total_h), (17, 19, 24))
    draw = ImageDraw.Draw(sheet)
    title, caption = font("SemiBold", 22), font("Regular", 18)

    for index, (panel, (heading, detail)) in enumerate(zip(panels, captions)):
        x = PAD + index * (width + GAP)
        centred(draw, (x, PAD, x + width, PAD + BAND / 2), heading, title, (236, 240, 246))
        centred(draw, (x, PAD + BAND / 2, x + width, PAD + BAND), detail, caption, (150, 158, 170))
        sheet.paste(panel, (x, PAD + BAND))

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="build")
    parser.add_argument("--image", default="calib/dog.jpg")
    parser.add_argument("--prompt", default="dog")
    parser.add_argument("--size", type=int, default=672)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--out", default="assets/compare.png")
    args = parser.parse_args()

    model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float32).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    picture = Image.open(args.image).convert("RGB").resize((args.size, args.size))
    panels, captions, shapes = [], [], []

    for label, stem, colour in (
        ("float", "hidden_float", MASK_COLOURS[0]),
        ("INT8", "hidden_int8", MASK_COLOURS[1]),
    ):
        hidden = np.load(f"{args.build}/{stem}.npy")
        vision = vision_output(model, hidden, args.size)
        masks, scores = masks_of(
            model, processor, vision, args.prompt, picture.size[::-1], args.threshold
        )
        pixels = int(sum(m.sum() for m in masks))
        print(f"{label}: {len(masks)} instances, {pixels} px, scores {scores}", flush=True)

        shapes.append(masks)
        panels.append(overlay(picture, masks, colour))
        captions.append((
            "ONNX (CPU, float)" if label == "float" else "DXNN (INT8, DX-COM scales)",
            f"{len(masks)} instance, {pixels:,} px",
        ))

    union = [np.logical_or.reduce(m) if m else None for m in shapes]
    if all(u is not None for u in union):
        overlap = np.logical_and(*union).sum() / np.logical_or(*union).sum()
        print(f"mask IoU {overlap:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    compose(panels, captions).save(args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
