"""Crop the window title bar off full-screen captures and save 16:9 JPEGs.

Usage: python tools/ingame/crop_screenshots.py RAW_DIR OUT_DIR [prefix]
Files named <prefix><view>.png become OUT_DIR/<view>.jpg.
"""
import sys
from pathlib import Path

from PIL import Image

TITLE_BAR = 36  # px of window title bar (and a margin) at the top of a maximized-window capture


def main(raw_dir: str, out_dir: str, prefix: str = "") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(raw_dir).glob(f"{prefix}*.png")):
        with Image.open(f) as im:
            w = im.width
            h = min(im.height - TITLE_BAR, round(w * 9 / 16))
            crop = im.convert("RGB").crop((0, TITLE_BAR, w, TITLE_BAR + h))
        dest = out / (f.stem[len(prefix):] + ".jpg")
        crop.save(dest, "JPEG", quality=88)
        print(dest, crop.size)


if __name__ == "__main__":
    main(*sys.argv[1:])
