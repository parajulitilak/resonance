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
DATA = (201, 209, 220)        # measured spectrum
MODEL = (86, 180, 233)        # the fitted ladder (Okabe-Ito sky blue)
TEAL = (38, 196, 170)         # verified
AMBER = (230, 159, 0)         # provisional (Okabe-Ito orange)
RED = (255, 106, 61)          # refused (vermilion)
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


def running_locks(tr: Tracking, report: dict, every_s: float = 0.2) -> dict:
    """For each member: the running ladder fit over time, and the frame where it locks.

    Lock = the first checkpoint from which the running fit (data seen so far only) is verified or
    provisional AND agrees with the final f0 at every later checkpoint. Nothing here is scripted.
    """
    from .spectrum import fit_harmonics, peaks, spectrum

    t = tr.t
    fs = tr.timebase.fps
    step = max(1, int(round(every_s * fs)))
    ks = list(range(max(step, int(0.5 * fs)), tr.n_frames + 1, step))
    if not ks or ks[-1] != tr.n_frames:
        ks.append(tr.n_frames)
    out = {}
    for e, m in zip(report["members"], tr.members):
        final = e["ladder"]
        prior = tuple(e["prior_f0_hz"]) if e.get("prior_f0_hz") else None
        w = tr.weights[m.name] if tr.weights else None
        rows = []
        for k in ks:
            sp = spectrum(t[:k], tr.disp[m.name][:k], weights=w)
            fit = fit_harmonics(peaks(sp), sp.fs, f0_range=prior, duration=float(t[k - 1] - t[0]))
            agree = (final["status"] != "refused" and fit.status != "refused" and np.isfinite(fit.f0)
                     and abs(fit.f0 - final["f0_hz"]) <= max(3 * final["f0_sigma_hz"], 0.005 * final["f0_hz"]))
            rows.append((k, fit.status, fit.f0, [n for n, _, _ in fit.modes], agree))
        lock = None
        for i in range(len(rows) - 1, -1, -1):
            if rows[i][4]:
                lock = rows[i][0]
            else:
                break
        out[m.name] = {"checkpoints": rows, "lock_frame": lock}
    return out


def render_hero(video: str, tr: Tracking, report: dict, out: str | Path, speed: float = 0.25,
                out_fps: float = 30.0, magnify: float = 25.0, f_max: float | None = None,
                title: str = "RESONANCE", subtitle: str = "cable tension from a phone video") -> Path:
    """`speed` < 1 plays in slow motion (0.25 = 4x slower than real time). Every state shown is computed
    from the frames seen so far (see `running_locks`)."""
    out = Path(out)
    t = tr.t
    fs = tr.timebase.fps
    step = max(1, int(round(fs * speed / out_fps)))
    f_max = f_max or min(fs / 2, 120.0)
    by = {m["name"]: m for m in report["members"]}
    members = tr.members
    locks = running_locks(tr, report)

    def state_at(name: str, k: int):
        rows = locks[name]["checkpoints"]
        cur = None
        for r in rows:
            if r[0] <= k + 1:
                cur = r
        return cur

    info = probe(video)
    vh, vw = info.height, info.width
    scale = min(OUT_H / vh, 0.55 * OUT_W / vw)    # portrait fills the height; landscape takes 55% of the width
    vid_w, vid_h = int(round(vw * scale)), int(round(vh * scale))
    oy = (OUT_H - vid_h) // 2
    px = vid_w + 40
    pw = OUT_W - px - 40
    # at most six rows in the panel (measured members first); the rest are summarised in one line
    rank = {"verified": 0, "provisional": 1, "frequency": 2, "refused": 3}
    panel = sorted(members, key=lambda m: rank.get(by[m.name]["ladder"]["status"], 4))[:6]
    hidden = [m for m in members if m not in panel]
    row_h = min((OUT_H - 170 - (40 if hidden else 0)) / len(panel), 300.0)  # one cable: not a tall thin strip
    f_title, f_sub, f_name, f_small = _font(40, True), _font(22), _font(26, True), _font(19)
    f_chip, f_big = _font(24, True), _font(40, True)

    with VideoWriter(str(out), OUT_W, OUT_H, out_fps, crf=18) as vw_:
        k_last = ((tr.n_frames - 1) // step) * step   # the last frame we will draw
        hold = int(1.5 * out_fps)  # after the take ends, hold the final verdicts long enough to read
        for k, frame in enumerate(read_frames(video)):
            if k % step or k > k_last:
                continue
            last = k == k_last
            canvas = Image.new("RGB", (OUT_W, OUT_H), BG)
            canvas.paste(Image.fromarray(frame).resize((vid_w, vid_h), Image.BILINEAR), (0, oy))
            d = ImageDraw.Draw(canvas, "RGBA")
            for m in members:
                lk = locks[m.name]["lock_frame"]
                st = by[m.name]["ladder"]["status"]
                locked = lk is not None and k + 1 >= lk
                col = STATUS_COLOUR[st] if locked or (last and st == "refused") else DATA
                u = tr.disp[m.name][k] - tr.disp[m.name][: max(k, 1)].mean(0)
                pts = []
                for s_, du in zip(tr.stations, u):
                    x = m.p0[0] + s_ * (m.p1[0] - m.p0[0]) + magnify * du * m.normal[0]
                    y = m.p0[1] + s_ * (m.p1[1] - m.p0[1]) + magnify * du * m.normal[1]
                    pts.append((x * scale, y * scale + oy))
                d.line(pts, fill=col + (240,), width=6)
                d.text((m.p0[0] * scale - 18, m.p0[1] * scale - 34 + oy), m.name, font=f_small, fill=INK)
            d.rounded_rectangle([16, OUT_H - 52, 210, OUT_H - 16], radius=8, fill=(0, 0, 0, 150))
            d.text((28, OUT_H - 47), f"motion ×{magnify:.0f}", font=f_sub, fill=INK)

            d.rectangle([px - 20, 0, OUT_W, OUT_H], fill=PANEL)
            d.text((px, 28), title, font=f_title, fill=INK)
            d.text((px, 82), f"{subtitle}  ·  {fs:.0f} fps, shown {1 / speed:.0f}× slower", font=f_sub, fill=DIM)
            if hidden and last:
                sts = sorted({by[m.name]["ladder"]["status"] for m in hidden})
                d.text((px, OUT_H - 52), f"+ {len(hidden)} more: {', '.join(sts)}", font=f_sub, fill=DIM)
            for i, m in enumerate(panel):
                e = by[m.name]
                lad = e["ladder"]
                st = lad["status"]
                cur = state_at(m.name, k)
                lk = locks[m.name]["lock_frame"]
                locked = lk is not None and k + 1 >= lk
                y0 = 150 + i * row_h
                gx0, gx1, gy0, gy1 = px + 150, px + pw - 250, y0 + 18, y0 + row_h - 34
                d.text((px, y0 + 18), m.name, font=f_name, fill=INK)
                spec = _rolling_spectrum(t, tr.disp[m.name], k + 1, f_max)
                d.rectangle([gx0, gy0, gx1, gy1], outline=(60, 66, 78), width=1)
                xs = np.linspace(gx0, gx1, spec.size)
                ys = gy1 - spec * (gy1 - gy0)
                d.polygon([(gx0, gy1)] + list(zip(xs, ys)) + [(gx1, gy1)], fill=DATA + (50,))
                d.line(list(zip(xs, ys)), fill=DATA, width=2)
                raw_step = f_max / 6
                mag = 10 ** math.floor(math.log10(raw_step))
                tstep = next(k * mag for k in (1, 2, 5, 10) if k * mag >= raw_step)
                ticks = [round(k * tstep, 6) for k in range(int(f_max / tstep) + 1)]
                for hz in ticks:
                    x = gx0 + hz / f_max * (gx1 - gx0)
                    lab = f"{hz:g}"
                    label = f"{lab} Hz" if hz == ticks[-1] else lab
                    d.text((x - (d.textlength(label, font=f_small) if hz == ticks[-1] else 8), gy1 + 4), label,
                           font=f_small, fill=DIM)
                # rungs the running fit has matched so far (model colour, dashed)
                if cur and cur[1] != "refused" and np.isfinite(cur[2]):
                    f0c, matched = (lad["f0_hz"], [mm["n"] for mm in lad["modes"]]) if locked else (cur[2], cur[3])
                    bb = lad["b"] if locked else 0.0
                    for nn in matched:
                        fn = nn * f0c * math.sqrt(1 + bb * nn * nn)
                        if fn > f_max:
                            continue
                        x = gx0 + fn / f_max * (gx1 - gx0)
                        for yy in range(int(gy0), int(gy1), 10):
                            d.line([(x, yy), (x, min(yy + 5, gy1))], fill=MODEL + (220,), width=2)
                        d.text((x + 3, gy0 + 2), str(nn), font=f_small, fill=MODEL)
                rx = gx1 + 22
                if locked:
                    chip = st.upper()
                    tw = d.textlength(chip, font=f_chip)
                    d.rounded_rectangle([rx, y0 + 16, rx + tw + 24, y0 + 50], radius=16,
                                        fill=STATUS_COLOUR[st] + (45,))
                    d.text((rx + 12, y0 + 19), chip, font=f_chip, fill=STATUS_COLOUR[st])
                    ten = e.get("tension") or {}
                    if ten.get("value_n") is not None:
                        d.text((rx, y0 + 58), f"{ten['value_n']:.1f} N", font=f_big, fill=INK)
                        d.text((rx, y0 + 104), f"± {ten['sigma_n']:.1f} N · f0 {lad['f0_hz']:.2f} Hz",
                               font=f_small, fill=DIM)
                    else:
                        d.text((rx, y0 + 60), f"f0 {lad['f0_hz']:.2f} Hz", font=f_sub, fill=INK)
                elif last and st == "refused":
                    tw = d.textlength("REFUSED", font=f_chip)
                    d.rounded_rectangle([rx, y0 + 16, rx + tw + 24, y0 + 50], radius=16, fill=RED + (45,))
                    d.text((rx + 12, y0 + 19), "REFUSED", font=f_chip, fill=RED)
                    d.text((rx, y0 + 60), (lad["reasons"] or ["no harmonic ladder"])[0][:34] + "…",
                           font=f_small, fill=DIM)
                else:
                    n_r = len(cur[3]) if cur and cur[1] != "refused" else 0
                    d.text((rx, y0 + 24), "listening…", font=f_sub, fill=DIM)
                    d.text((rx, y0 + 54), f"{n_r} rung{'s' if n_r != 1 else ''} matched", font=f_small, fill=DIM)
            arr = np.asarray(canvas)
            vw_.write(arr)
            if last:
                for _ in range(hold):
                    vw_.write(arr)
    return out
