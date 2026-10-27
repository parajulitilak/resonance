"""Synthetic bench rig with known tensions, rendered like a 240 fps phone clip.

Five ropes hang from a beam, each loaded with a jug of known mass (T = M g).
Each rope is plucked once and rings with its true harmonic ladder
f_n = n f0 sqrt(1 + B n^2), amplitudes from a triangular pluck. The camera
shakes slightly; an anchor card on the wall lets the pipeline remove it.
Every number the pipeline should recover is written to <out>.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .physics import G
from .video import VideoWriter

W, H = 720, 1280            # portrait, like a phone held upright
SS = 2
SCALE = 560.0               # pixels per metre at the rope plane
BEAM_Y = 110


@dataclass
class Rope:
    name: str
    x: float          # pixel column
    length_m: float
    mass_kg: float
    mass_per_m: float = 0.022
    b: float = 2e-5
    pluck_t: float = 0.4
    pluck_at: float = 0.3
    amp_px: float = 4.0
    tau: float = 1.6

    @property
    def tension(self) -> float:
        return self.mass_kg * G

    @property
    def f0(self) -> float:
        return math.sqrt(self.tension / self.mass_per_m) / (2 * self.length_m)

    @property
    def bottom_y(self) -> float:
        return BEAM_Y + self.length_m * SCALE

    def modes(self, f_max: float = 400.0, n_max: int = 40):
        """Physical modes up to f_max, independent of the camera (so low frame rates alias, as in reality)."""
        out = []
        for n in range(1, n_max + 1):
            fn = n * self.f0 * math.sqrt(1 + self.b * n * n)
            a = math.sin(n * math.pi * self.pluck_at) / n ** 2
            if fn < f_max and abs(a) > 1e-3:
                out.append((n, fn, a))
        return out


def default_rig() -> list[Rope]:
    masses = [2.0, 4.0, 6.0, 8.0, 10.0]
    lengths = [1.78, 1.74, 1.81, 1.70, 1.76]
    return [Rope(f"rope{i + 1}", 120 + 120 * i, lengths[i], masses[i], pluck_t=0.3 + 0.12 * i,
                 pluck_at=0.27 + 0.05 * i) for i in range(5)]


def _background(rng: np.random.Generator) -> Image.Image:
    yy, xx = np.mgrid[0:H * SS, 0:W * SS]
    blot = sum(np.sin(xx / (40 + 30 * k) + k) * np.cos(yy / (55 + 25 * k) - k) for k in range(3))
    base = np.stack([228 + 4 * blot, 222 + 4 * blot, 210 + 4 * blot], -1)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([0, (BEAM_Y - 34) * SS, W * SS, (BEAM_Y + 6) * SS], fill=(110, 78, 48))
    for k in range(0, W, 23):  # wood grain
        d.line([k * SS, (BEAM_Y - 30) * SS, (k + 9) * SS, (BEAM_Y + 2) * SS], fill=(96, 66, 40), width=SS)
    cx, cy, s = 40, BEAM_Y + 40, 12  # anchor card: a checkerboard taped to the wall
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                d.rectangle([(cx + i * s) * SS, (cy + j * s) * SS, (cx + (i + 1) * s) * SS, (cy + (j + 1) * s) * SS],
                            fill=(20, 20, 20))
    return img


ANCHOR_BOX = (24, BEAM_Y + 24, 104, BEAM_Y + 104)  # x0, y0, x1, y1 around the anchor card


def render(out: str | Path, fps: float = 240.0, duration: float = 5.0, ropes: list[Rope] | None = None,
           noise: float = 2.0, shake_px: float = 0.6, seed: int = 0, crf: int = 16) -> dict:
    rng = np.random.default_rng(seed)
    ropes = ropes or default_rig()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bg = _background(rng)
    n = int(round(duration * fps))
    # smooth camera shake: low-passed random walk
    walk = np.cumsum(rng.normal(0, 1, (n, 2)), 0)
    walk -= np.linspace(walk[0], walk[-1], n)
    walk = shake_px * walk / (np.abs(walk).max() + 1e-9)
    modes = {r.name: r.modes() for r in ropes}
    s = np.linspace(0, 1, 60)
    with VideoWriter(str(out), W, H, fps, crf=crf) as vw:
        for i in range(n):
            t = i / fps
            im = bg.copy()
            d = ImageDraw.Draw(im)
            for r in ropes:
                u = np.zeros_like(s)
                if t >= r.pluck_t:
                    tt = t - r.pluck_t
                    for k, fn, a in modes[r.name]:
                        u += r.amp_px * a / 0.9 * math.exp(-tt * math.sqrt(k) / r.tau) * np.sin(k * math.pi * s) \
                             * math.cos(2 * math.pi * fn * tt)
                ys = BEAM_Y + s * (r.bottom_y - BEAM_Y)
                pts = [((r.x + uu) * SS, yy * SS) for uu, yy in zip(u, ys)]
                d.line(pts, fill=(58, 58, 62), width=int(3.4 * SS))
                jx, jy = r.x, r.bottom_y
                d.rectangle([(jx - 34) * SS, jy * SS, (jx + 34) * SS, (jy + 110) * SS], fill=(70, 120, 200))
                d.rectangle([(jx - 8) * SS, (jy - 10) * SS, (jx + 8) * SS, (jy + 2) * SS], fill=(40, 40, 40))
            frame = im.resize((W, H), Image.BOX)
            dx, dy = walk[i]
            frame = frame.transform((W, H), Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.BILINEAR)
            arr = np.asarray(frame, np.float32) + rng.normal(0, noise, (H, W, 3))
            vw.write(np.clip(arr, 0, 255).astype(np.uint8))
    truth = {
        "fps": fps, "frames": n, "scale_px_per_m": SCALE, "anchor_box": ANCHOR_BOX,
        "ropes": [{"name": r.name, "p0": [r.x, BEAM_Y], "p1": [r.x, r.bottom_y], "length_m": r.length_m,
                   "mass_per_m": r.mass_per_m, "mass_kg": r.mass_kg, "tension_n": r.tension, "f0_hz": r.f0,
                   "b": r.b, "pluck_t": r.pluck_t} for r in ropes],
    }
    Path(str(out) + ".json").write_text(json.dumps(truth, indent=2))
    return truth
