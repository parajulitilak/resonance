"""Synthetic footbridge seen from the bank: towers, a sagging main cable, hangers and a deck.

The main cable sways in its first antisymmetric mode at sqrt(g / 8b) and a
symmetric mode, with slowly varying amplitude (people walking). Ground truth:
span, sag and both frequencies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .physics import sag_frequency
from .video import VideoWriter

W, H, SS = 1280, 720, 2


def render(out: str | Path, span_m: float = 30.0, sag_m: float = 1.5, fps: float = 60.0, duration: float = 14.0,
           scale: float = 36.0, f_sym: float = 1.37, noise: float = 2.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    xl, xr, ytop = 100.0, 100.0 + span_m * scale, 200.0
    sag_px = sag_m * scale
    deck_y = 470.0
    f_anti = sag_frequency(sag_m)
    base = Image.new("RGB", (W * SS, H * SS))
    d = ImageDraw.Draw(base)
    for y in range(H * SS):  # sky gradient
        c = int(205 - 60 * y / (H * SS))
        d.line([(0, y), (W * SS, y)], fill=(c - 30, c - 5, min(255, c + 30)))
    for k in range(40):  # tree line
        cx, r = rng.uniform(0, W), rng.uniform(25, 60)
        d.ellipse([(cx - r) * SS, (560 - r) * SS, (cx + r) * SS, (560 + r) * SS], fill=(40, 70, 45))
    d.rectangle([0, 580 * SS, W * SS, H * SS], fill=(90, 110, 120))  # river
    for x in (xl, xr):
        d.rectangle([(x - 9) * SS, (ytop - 6) * SS, (x + 9) * SS, (deck_y + 90) * SS], fill=(150, 150, 150))
    n = int(round(duration * fps))
    t_axis = np.arange(n) / fps
    env_a = 1.2 + 0.4 * np.sin(2 * math.pi * 0.07 * t_axis + 1.0)
    env_s = 0.7 + 0.3 * np.sin(2 * math.pi * 0.05 * t_axis)
    xs = np.linspace(xl, xr, 180)
    u = (xs - xl) / (xr - xl)
    with VideoWriter(str(out), W, H, fps, crf=18) as vw:
        for i, t in enumerate(t_axis):
            im = base.copy()
            dd = ImageDraw.Draw(im)
            dy = env_a[i] * np.sin(2 * math.pi * u) * math.sin(2 * math.pi * f_anti * t) \
                + env_s[i] * np.sin(math.pi * u) * math.sin(2 * math.pi * f_sym * t + 0.7)
            ys = ytop + 4 * sag_px * u * (1 - u) + dy
            dd.line([(x * SS, y * SS) for x, y in zip(xs, ys)], fill=(30, 30, 34), width=3 * SS)
            for hx in np.arange(xl + scale, xr - 1, scale):
                hy = np.interp(hx, xs, ys)
                dd.line([(hx * SS, hy * SS), (hx * SS, deck_y * SS)], fill=(60, 60, 64), width=SS)
            dd.rectangle([xl * SS, deck_y * SS, xr * SS, (deck_y + 6) * SS], fill=(70, 60, 50))
            arr = np.asarray(im.resize((W, H), Image.BOX), np.float32) + rng.normal(0, noise, (H, W, 3))
            vw.write(np.clip(arr, 0, 255).astype(np.uint8))
    truth = {"span_m": span_m, "sag_m": sag_m, "f_antisymmetric_hz": f_anti, "f_symmetric_hz": f_sym,
             "scale_px_per_m": scale, "fps": fps, "frames": n,
             "cable_box": [int(xl) + 12, int(ytop) - 6, int(xr) - 12, int(ytop + sag_px) + 8]}
    Path(str(out) + ".json").write_text(json.dumps(truth, indent=2))
    return truth
