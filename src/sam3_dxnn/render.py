import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAD, BAND, GAP = 18, 62, 14
BACKGROUND = (17, 19, 24)
TITLE, DETAIL = (236, 240, 246), (150, 158, 170)


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    name = f"Pretendard-{weight}.otf"
    # Pillow's own font search skips per-user installs on Windows
    user = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts", name)

    return ImageFont.truetype(user if os.path.exists(user) else name, size)


def fit(image: Image.Image, size: int) -> Image.Image:
    scale = size / max(image.size)

    return image.convert("RGB").resize(
        (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
    )


def overlay(image: Image.Image, masks: list[np.ndarray], colour) -> Image.Image:
    masks = [np.asarray(Image.fromarray(m).resize(image.size, Image.NEAREST)) for m in masks]
    canvas = np.asarray(image, np.float32)

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


def compose(panels: list[Image.Image], captions: list[tuple[str, str]]) -> Image.Image:
    width, height = panels[0].size
    sheet = Image.new("RGB", (PAD * 2 + width * 2 + GAP, PAD * 2 + BAND + height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    title, caption = font("SemiBold", 22), font("Regular", 18)

    for index, (panel, (heading, detail)) in enumerate(zip(panels, captions)):
        x = PAD + index * (width + GAP)
        centred(draw, (x, PAD, x + width, PAD + BAND / 2), heading, title, TITLE)
        centred(draw, (x, PAD + BAND / 2, x + width, PAD + BAND), detail, caption, DETAIL)
        sheet.paste(panel, (x, PAD + BAND))

    return sheet
