#!/usr/bin/env python3
"""Print an ArUco marker at an exact physical size.

    python3 scripts/make_tag.py --id 0 --size 0.05 --out /tmp/tag0.png

The printed EDGE LENGTH of the black square must match `marker_size` on
the tag_follow node to the millimetre — pose error scales directly with
it. Print at 100% / "actual size" (no fit-to-page), then measure the
black square with a ruler and pass what you measured.
"""

import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--size", type=float, default=0.05, help="metres")
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="tag.png")
    a = ap.parse_args()

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dictionary))
    px = int(round(a.size / 0.0254 * a.dpi))          # marker edge in pixels
    marker = cv2.aruco.generateImageMarker(d, a.id, px)
    quiet = max(px // 5, 20)                           # white border, >= 1 cell
    canvas = np.full((px + 2 * quiet, px + 2 * quiet), 255, np.uint8)
    canvas[quiet:quiet + px, quiet:quiet + px] = marker
    canvas = cv2.copyMakeBorder(canvas, 0, 90, 0, 0, cv2.BORDER_CONSTANT,
                                value=255)
    cv2.putText(canvas, "%s  id=%d  edge=%.0f mm" % (a.dictionary, a.id,
                                                     a.size * 1000),
                (quiet, canvas.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, 0, 2)
    cv2.imwrite(a.out, canvas)
    print("wrote %s — print at 100%%, black square should measure %.0f mm"
          % (a.out, a.size * 1000))


if __name__ == "__main__":
    main()
