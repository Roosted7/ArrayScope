#!/usr/bin/env python3
"""Generate the ArrayScope application icon assets.

Draws the icon programmatically (a 3x3 array heatmap under a magnifier
ring) and writes PNGs at the standard sizes plus a multi-resolution
Windows .ico into ``arrayscope/resources/icons/``. The scalable SVG
(``arrayscope.svg``) is maintained by hand to match this design.

Run from the repository root after changing the design:

    python tools/generate_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "arrayscope" / "resources" / "icons"
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)

BACKGROUND = (16, 20, 31, 255)  # deep navy
# 3x3 heatmap tiles, viridis-flavoured, brightest around the center — reads
# as "array with structure" even at 16 px.
TILE_COLORS = [
    [(68, 1, 84), (59, 82, 139), (33, 145, 140)],
    [(49, 104, 142), (253, 231, 37), (94, 201, 98)],
    [(33, 145, 140), (144, 215, 67), (68, 1, 84)],
]
RING = (235, 240, 248, 255)


def draw_icon(size):
    scale = 8  # supersample for crisp edges at small sizes
    canvas = size * scale
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    radius = canvas * 0.18
    draw.rounded_rectangle((0, 0, canvas - 1, canvas - 1), radius=radius, fill=BACKGROUND)

    # 3x3 tile grid
    margin = canvas * 0.14
    gap = canvas * 0.045
    cell = (canvas - 2 * margin - 2 * gap) / 3
    tile_radius = cell * 0.18
    for row in range(3):
        for col in range(3):
            x0 = margin + col * (cell + gap)
            y0 = margin + row * (cell + gap)
            draw.rounded_rectangle(
                (x0, y0, x0 + cell, y0 + cell),
                radius=tile_radius,
                fill=TILE_COLORS[row][col] + (255,),
            )

    # Magnifier ring over the lower-right tiles
    ring_center = (canvas * 0.62, canvas * 0.62)
    ring_radius = canvas * 0.26
    ring_width = max(2 * scale, int(canvas * 0.055))
    bbox = (
        ring_center[0] - ring_radius,
        ring_center[1] - ring_radius,
        ring_center[0] + ring_radius,
        ring_center[1] + ring_radius,
    )
    draw.ellipse(bbox, outline=RING, width=ring_width)
    # Handle: 45° stroke from the ring edge toward the corner
    handle_start = (
        ring_center[0] + ring_radius * 0.707,
        ring_center[1] + ring_radius * 0.707,
    )
    handle_end = (canvas * 0.94, canvas * 0.94)
    draw.line([handle_start, handle_end], fill=RING, width=int(ring_width * 1.25))

    return image.resize((size, size), Image.LANCZOS)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = {}
    for size in SIZES:
        image = draw_icon(size)
        images[size] = image
        path = OUTPUT_DIR / f"arrayscope-{size}.png"
        image.save(path)
        print(f"wrote {path}")
    ico_path = OUTPUT_DIR / "arrayscope.ico"
    images[256].save(
        ico_path,
        sizes=[(s, s) for s in SIZES if s <= 256],
        append_images=[images[s] for s in SIZES if s <= 256],
    )
    print(f"wrote {ico_path}")


if __name__ == "__main__":
    main()
