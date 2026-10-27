"""Field mode for footbridges: main-cable sag and its predicted sway frequency.

A shallow suspension cable's first antisymmetric in-plane mode has frequency
f = sqrt(g / (8 b)) for mid-span sag b, whatever its length or weight. So the
video gives two independent numbers for the same cable: the sag (geometry,
from the cable's shape in one frame) and the frequency (dynamics, from its
motion). If they disagree, something about the cable or the measurement is off.
"""

from __future__ import annotations

from dataclasses import dataclass

import subprocess

import numpy as np
from scipy.ndimage import map_coordinates

from .displacement import _centroids
from .physics import sag_frequency
from .spectrum import mode_shape, peaks, spectrum
from .timing import frame_times
from .video import probe, read_frames


def first_frame(video: str) -> np.ndarray:
    info = probe(video)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8)[: info.width * info.height].reshape(info.height, info.width)


@dataclass
class CableProfile:
    x: np.ndarray            # columns sampled
    y: np.ndarray            # sub-pixel cable row per column
    coef: np.ndarray         # parabola y = c0 x^2 + c1 x + c2
    x_left: float
    x_right: float
    sag_px: float            # max depth below the chord between the two ends
    rms_px: float            # fit residual

    def y_at(self, x):
        return np.polyval(self.coef, x)


def fit_profile(gray: np.ndarray, box: tuple[int, int, int, int], dark: bool = True,
                band: int = 6) -> CableProfile:
    """Trace the cable column by column inside its box and fit a parabola."""
    x1, y1, x2, y2 = box
    xs = np.arange(x1, x2 + 1, 2, dtype=float)
    sub = gray[y1:y2 + 1, :].astype(np.float32)
    ys = []
    for x in xs.astype(int):
        col = sub[:, x]
        k = int(np.argmin(col) if dark else np.argmax(col))
        lo, hi = max(0, k - band), min(col.size, k + band + 1)
        w = (col[lo:hi].max() - col[lo:hi]) if dark else (col[lo:hi] - col[lo:hi].min())
        ys.append(y1 + lo + float((w * np.arange(hi - lo)).sum() / max(w.sum(), 1e-6)))
    ys = np.array(ys)
    coef = np.polyfit(xs, ys, 2)
    for _ in range(2):  # drop outliers (trees, hangers) and refit
        r = ys - np.polyval(coef, xs)
        keep = np.abs(r) < 3 * max(np.median(np.abs(r)) * 1.4826, 0.3)
        coef = np.polyfit(xs[keep], ys[keep], 2)
    xl, xr = float(xs[0]), float(xs[-1])
    yl, yr = np.polyval(coef, [xl, xr])
    xm = np.linspace(xl, xr, 400)
    chord = yl + (yr - yl) * (xm - xl) / (xr - xl)
    sag = float(np.max(np.polyval(coef, xm) - chord))
    rms = float(np.sqrt(np.mean((ys - np.polyval(coef, xs)) ** 2)))
    return CableProfile(xs, ys, coef, xl, xr, sag, rms)


@dataclass
class SagCheck:
    sag_m: float
    sag_ratio: float          # sag / span
    predicted_hz: float
    measured_hz: float | None
    agreement: float | None   # measured / predicted
    antisymmetric: bool | None
    note: str


def track_cable(video: str, prof: CableProfile, n_stations: int = 24, half_width: float = 10.0,
                max_frames: int | None = None):
    """Vertical-ish displacement along the fitted curve (normal to the local tangent)."""
    s = np.linspace(0.08, 0.92, n_stations)
    xs = prof.x_left + s * (prof.x_right - prof.x_left)
    ys = prof.y_at(xs)
    slope = np.polyval(np.polyder(prof.coef), xs)
    nx, ny = -slope, np.ones_like(slope)
    norm = np.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    o = np.arange(-half_width, half_width + 1e-9, 0.5)
    gx = xs[:, None] + o[None, :] * nx[:, None]
    gy = ys[:, None] + o[None, :] * ny[:, None]
    out = []
    for i, frame in enumerate(read_frames(video, gray=True)):
        if max_frames and i >= max_frames:
            break
        prof_ = map_coordinates(frame.astype(np.float32), [gy.ravel(), gx.ravel()], order=1,
                                mode="nearest").reshape(gx.shape)
        cen, _ = _centroids(prof_, o, -1.0)
        out.append(cen)
    return s, np.array(out)


def sag_check(video: str, box: tuple[int, int, int, int], span_m: float, capture_fps: float | None = None,
              towers_x: tuple[float, float] | None = None, f_band: tuple[float, float] = (0.2, 5.0)) -> SagCheck:
    """Sag from the first frame, sway frequency from the motion, and the physics cross-check.

    With `towers_x` (pixel columns where the cable meets the two towers, e.g. from SAM "bridge tower"
    boxes), sag comes from the fitted curvature over the full span: b = |c0| span_px^2 / 4. Without it,
    the chord between the box ends is used, which underestimates sag if the box is inset.
    """
    prof = fit_profile(first_frame(video).astype(np.float32), box)
    if towers_x:
        span_px = abs(towers_x[1] - towers_x[0])
        sag_px = abs(prof.coef[0]) * span_px ** 2 / 4
    else:
        span_px, sag_px = prof.x_right - prof.x_left, prof.sag_px
    scale = span_px / span_m                                 # px per metre along the span
    sag_m = sag_px / scale
    pred = sag_frequency(sag_m)
    tb = frame_times(video, capture_fps)
    s, disp = track_cable(video, prof)
    t = tb.t[: disp.shape[0]]
    sp = spectrum(t, disp, pad=8)
    pk = [p for p in peaks(sp, f_min=f_band[0], min_snr=6.0) if p.f <= f_band[1]]
    measured, anti, note = None, None, ""
    if pk:
        # the antisymmetric mode: node at mid-span, quarter spans move in opposite directions
        for p in sorted(pk, key=lambda p: abs(p.f - pred)):
            shp = mode_shape(sp, p.f)
            q1, q3 = shp[np.argmin(np.abs(s - 0.25))], shp[np.argmin(np.abs(s - 0.75))]
            mid = shp[np.argmin(np.abs(s - 0.5))]
            if q1 * q3 < 0 and abs(mid) < 0.5 * max(abs(q1), abs(q3)):
                measured, anti = p.f, True
                break
        if measured is None:
            measured, anti = min(pk, key=lambda p: abs(p.f - pred)).f, False
            note = "nearest peak is not antisymmetric; treat as unconfirmed"
    else:
        note = f"no clear sway peak between {f_band[0]} and {f_band[1]} Hz"
    agree = None if measured is None else measured / pred
    if agree is not None and not note:
        note = "consistent" if abs(agree - 1) < 0.15 else "sag and sway frequency disagree: inspect"
    return SagCheck(sag_m, sag_m / span_m, pred, measured, agree, anti, note)
