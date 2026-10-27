"""The hero shot: the video with each member's vibration magnified, beside a live spectrum
that builds up as the clip plays until the peaks lock onto the harmonic ladder.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .displacement import Tracking
from .spectrum import uniform
from .video import VideoWriter, probe, read_frames

OUT_W, OUT_H = 1920, 1080
BG = (14, 17, 22)
PANEL = (24, 28, 36)
INK = (232, 236, 242)
DIM = (130, 140, 155)
TEAL = (38, 196, 170)
AMBER = (255, 176, 46)
RED = (240, 84, 84)
STATUS_COLOUR = {"verified": TEAL, "provisional": AMBER, "refused": RED}


def _font(size: int, bold: bool = False):
    for path, idx in [("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
                      ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
                      ("/System/Library/Fonts/SFNS.ttf", 0)]:
        try:
            return ImageFont.truetype(path, size, index=idx)
        except OSError:
            continue
    return ImageFont.load_default()


def _rolling_spectrum(t: np.ndarray, x: np.ndarray, upto: int, f_max: float, n_bins: int = 360):
    """Station-summed power on 0..f_max from frames [0, upto); log-normalised to 0..1."""
    if upto < 48:
        return np.zeros(n_bins)
    xu, fs = uniform(t[:upto], x[:upto])
    xu = xu - xu.mean(0)
    win = np.hanning(xu.shape[0])[:, None]
    nfft = max(4096, int(2 ** np.ceil(np.log2(xu.shape[0])) * 2))
    p = (np.abs(np.fft.rfft(xu * win, n=nfft, axis=0)) ** 2).sum(1)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    grid = np.linspace(0, f_max, n_bins)
    lp = np.log10(np.interp(grid, f, p) + 1e-12)
    lo, hi = np.percentile(lp, 20), lp.max()
    return np.clip((lp - lo) / max(hi - lo, 1e-6), 0, 1)


def render_hero(video: str, tr: Tracking, report: dict, out: str | Path, speed: float = 0.25,
                out_fps: float = 30.0, magnify: float = 25.0, f_max: float | None = None,
                title: str = "RESONANCE", subtitle: str = "cable tension from a phone video") -> Path:
    """`speed` < 1 plays in slow motion (0.25 = 4x slower than real time)."""
    out = Path(out)
    t = tr.t
    fs = tr.timebase.fps
    step = max(1, int(round(fs * speed / out_fps)))
    f_max = f_max or min(fs / 2, 120.0)
    by = {m["name"]: m for m in report["members"]}
    members = tr.members
    n = len(members)

    info = probe(video)
    vh, vw = info.height, info.width
    scale = OUT_H / vh
    vid_w = int(round(vw * scale))
    px = vid_w + 40
    pw = OUT_W - px - 40
    row_h = (OUT_H - 170) / n
    f_title, f_sub, f_name, f_small, f_big = _font(44, True), _font(24), _font(26, True), _font(19), _font(30, True)

    with VideoWriter(str(out), OUT_W, OUT_H, out_fps, crf=18) as vw_:
        for k, frame in enumerate(read_frames(video)):
            if k % step or k >= tr.n_frames:
                continue
            canvas = Image.new("RGB", (OUT_W, OUT_H), BG)
            canvas.paste(Image.fromarray(frame).resize((vid_w, OUT_H), Image.BILINEAR), (0, 0))
            d = ImageDraw.Draw(canvas, "RGBA")
            for m in members:
                st = by[m.name]["ladder"]["status"]
                col = STATUS_COLOUR[st]
                u = tr.disp[m.name][k] - tr.disp[m.name][: max(k, 1)].mean(0)
                pts = []
                for s, du in zip(tr.stations, u):
                    x = m.p0[0] + s * (m.p1[0] - m.p0[0]) + magnify * du * m.normal[0]
                    y = m.p0[1] + s * (m.p1[1] - m.p0[1]) + magnify * du * m.normal[1]
                    pts.append((x * scale, y * scale))
                d.line(pts, fill=col + (230,), width=4)
                lx, ly = m.p0[0] * scale, m.p0[1] * scale
                d.text((lx - 18, ly - 34), m.name.replace("rope", "#"), font=f_small, fill=INK)

            d.rectangle([px - 20, 0, OUT_W, OUT_H], fill=PANEL)
            d.text((px, 28), title, font=f_title, fill=INK)
            d.text((px, 84), subtitle + f"   ·   {fs:.0f} fps, shown {1 / speed:.0f}x slower   ·   t = {t[k]:.2f} s",
                   font=f_sub, fill=DIM)
            for i, m in enumerate(members):
                e = by[m.name]
                lad = e["ladder"]
                st = lad["status"]
                y0 = 150 + i * row_h
                gx0, gx1, gy0, gy1 = px + 170, px + pw - 270, y0 + 18, y0 + row_h - 34
                d.text((px, y0 + 18), m.name, font=f_name, fill=INK)
                spec = _rolling_spectrum(t, tr.disp[m.name], k + 1, f_max)
                d.rectangle([gx0, gy0, gx1, gy1], outline=(60, 66, 78), width=1)
                xs = np.linspace(gx0, gx1, spec.size)
                ys = gy1 - spec * (gy1 - gy0)
                d.polygon([(gx0, gy1)] + list(zip(xs, ys)) + [(gx1, gy1)], fill=(90, 150, 255, 90))
                d.line(list(zip(xs, ys)), fill=(130, 180, 255), width=2)
                for hz in range(0, int(f_max) + 1, 20):
                    x = gx0 + hz / f_max * (gx1 - gx0)
                    d.text((x - 8, gy1 + 4), str(hz), font=f_small, fill=DIM)
                progress = (k + 1) / tr.n_frames
                locked = progress > 0.6 and st != "refused"
                if locked:
                    f0 = lad["f0_hz"]
                    for nn in range(1, 40):
                        fn = nn * f0 * math.sqrt(1 + lad["b"] * nn * nn)
                        if fn > f_max:
                            break
                        x = gx0 + fn / f_max * (gx1 - gx0)
                        d.line([(x, gy0), (x, gy1)], fill=AMBER + (170,), width=2)
                        if nn <= 6:
                            d.text((x + 3, gy0 + 2), str(nn), font=f_small, fill=AMBER)
                rx = gx1 + 22
                if progress < 0.6:
                    d.text((rx, y0 + 30), "listening…", font=f_sub, fill=DIM)
                elif st == "refused":
                    d.text((rx, y0 + 22), "REFUSED", font=f_big, fill=RED)
                    d.text((rx, y0 + 62), "no harmonic ladder", font=f_small, fill=DIM)
                else:
                    ten = e.get("tension") or {}
                    d.text((rx, y0 + 18), f"f0 {lad['f0_hz']:.2f} Hz", font=f_sub, fill=INK)
                    if ten.get("value_n") is not None:
                        d.text((rx, y0 + 48), f"T = {ten['value_n']:.1f} N", font=f_big,
                               fill=STATUS_COLOUR[st])
                        d.text((rx, y0 + 86), f"± {ten['sigma_n']:.1f} N  {st}", font=f_small, fill=DIM)
            vw_.write(np.asarray(canvas))
    return out
