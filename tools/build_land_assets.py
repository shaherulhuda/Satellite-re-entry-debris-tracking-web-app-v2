"""Build the offline Earth assets in data/ from Natural Earth 110m land (public domain).

Usage: python tools/build_land_assets.py ne_110m_land.geojson
Writes data/coastlines.json (lon/lat rings) and data/land_mask.npy (land fraction on a
1.5-degree vertex grid, uint8 0-255, shape (121, 241): lat -90..90, lon -180..180).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

STEP = 1.5
SCALE = 4  # raster pixels per degree
OUT = Path(__file__).resolve().parent.parent / "data"


def rings_of(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for poly in polys:
        for ring in poly:
            yield ring


def main(src):
    gj = json.load(open(src))
    rings = [r for f in gj["features"] for r in rings_of(f["geometry"])]
    OUT.mkdir(exist_ok=True)
    json.dump([[[round(x, 2), round(y, 2)] for x, y in r] for r in rings],
              open(OUT / "coastlines.json", "w"), separators=(",", ":"))

    w, h = 360 * SCALE, 180 * SCALE
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for r in rings:
        d.polygon([((x + 180) * SCALE, (90 - y) * SCALE) for x, y in r], fill=255, outline=255)
    a = np.asarray(img, dtype=np.float32) / 255.0
    # average over a window of one grid step around each vertex
    win = int(STEP * SCALE)
    nlat, nlon = int(180 / STEP) + 1, int(360 / STEP) + 1
    mask = np.zeros((nlat, nlon), dtype=np.uint8)
    for i in range(nlat):
        row = int(round((90 - (-90 + i * STEP)) * SCALE))
        r0, r1 = max(0, row - win // 2), min(h, row + win // 2 + 1)
        for j in range(nlon):
            col = int(round(j * STEP * SCALE))
            cols = np.arange(col - win // 2, col + win // 2 + 1) % w
            mask[i, j] = int(round(255 * a[r0:r1][:, cols].mean()))
    np.save(OUT / "land_mask.npy", mask)
    print(len(rings), "rings; land fraction", mask.mean() / 255)


if __name__ == "__main__":
    main(sys.argv[1])
