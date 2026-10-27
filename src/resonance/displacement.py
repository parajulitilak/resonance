"""Sub-pixel transverse displacement of each member, frame by frame.

At every station the member shows up as a dip (dark rope on a light wall) or a
bump (bright cable against dark trees) in the cross profile. Its contrast-
weighted centroid gives the position to a small fraction of a pixel. Camera
shake is measured on a static anchor region by phase correlation and removed
along each member's normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .members import Member
from .timing import TimeBase, frame_times
from .video import read_frames


@dataclass
class Tracking:
    timebase: TimeBase
    members: list[Member]
    stations: np.ndarray                 # fractions along each member
    disp: dict[str, np.ndarray]          # name -> [frames, stations] pixels along the normal
    quality: dict[str, dict]             # name -> {"contrast": grey levels, "noise_px": position noise}
    shake: np.ndarray | None             # [frames, 2] anchor shift in pixels (x, y)

    @property
    def t(self) -> np.ndarray:
        return self.timebase.t[: self.n_frames]

    @property
    def n_frames(self) -> int:
        return next(iter(self.disp.values())).shape[0]


def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _refine(r: np.ndarray, y0: float, x0: float, factor: int, span: float) -> tuple[float, float]:
    """Evaluate the inverse DFT of `r` on a fine grid around (y0, x0) (Guizar-Sicairos et al. 2008)."""
    h, w = r.shape
    n = int(np.ceil(span * factor)) | 1
    ys = y0 + (np.arange(n) - n // 2) / factor
    xs = x0 + (np.arange(n) - n // 2) / factor
    c = (np.exp(2j * np.pi * np.outer(ys, np.fft.fftfreq(h))) @ r
         @ np.exp(2j * np.pi * np.outer(np.fft.fftfreq(w), xs))).real
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    return float(ys[iy]), float(xs[ix])


def phase_shift(ref_fft: np.ndarray, img: np.ndarray, win: np.ndarray) -> tuple[float, float]:
    """Shift (dx, dy) of `img` relative to the reference, to ~0.01 px (upsampled phase correlation)."""
    f = np.fft.fft2((img - img.mean()) * win)
    r = f * np.conj(ref_fft)
    r /= np.abs(r) + 1e-9
    c = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    h, w = c.shape
    y0 = iy - h if iy > h / 2 else iy
    x0 = ix - w if ix > w / 2 else ix
    y0, x0 = _refine(r, y0, x0, factor=10, span=1.5)
    dy, dx = _refine(r, y0, x0, factor=100, span=0.15)
    return dx, dy


def _centroids(profile: np.ndarray, offsets: np.ndarray, polarity: float) -> tuple[np.ndarray, np.ndarray]:
    """Contrast-weighted centroid per station; also returns peak contrast per station."""
    j = profile.shape[1]
    edge = max(2, j // 5)
    bg = np.median(np.concatenate([profile[:, :edge], profile[:, -edge:]], 1), axis=1, keepdims=True)
    c = np.clip(polarity * (profile - bg), 0, None)
    peak = c.max(axis=1, keepdims=True)
    c = np.clip(c - 0.25 * peak, 0, None)
    w = c.sum(1)
    return (c * offsets).sum(1) / np.maximum(w, 1e-9), peak[:, 0]


def track(path: str, members: list[Member], anchor: tuple[int, int, int, int] | None = None,
          n_stations: int = 32, capture_fps: float | None = None,
          max_frames: int | None = None) -> Tracking:
    """Track every member through the video. `anchor` is a static box (x0, y0, x1, y1)."""
    tb = frame_times(path, capture_fps)
    grids = {m.name: m.sample_grid(n_stations) for m in members}
    polarity: dict[str, float] = {}
    out: dict[str, list] = {m.name: [] for m in members}
    contrast: dict[str, list] = {m.name: [] for m in members}
    shake, ref_fft, win = [], None, None
    for i, frame in enumerate(read_frames(path, gray=True)):
        if max_frames and i >= max_frames:
            break
        f = frame.astype(np.float32)
        if anchor:
            x0, y0, x1, y1 = anchor
            crop = f[y0:y1, x0:x1]
            if ref_fft is None:
                win = _hann2d(*crop.shape)
                ref_fft = np.fft.fft2((crop - crop.mean()) * win)
            shake.append(phase_shift(ref_fft, crop, win))
        for m in members:
            xs, ys, _, o = grids[m.name]
            prof = map_coordinates(f, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)
            if m.name not in polarity:
                j = prof.shape[1]
                centre = np.median(prof[:, j // 2 - 1: j // 2 + 2])
                edges = np.median(np.concatenate([prof[:, :j // 5], prof[:, -(j // 5):]], 1))
                polarity[m.name] = -1.0 if centre < edges else 1.0
            cen, pk = _centroids(prof, o, polarity[m.name])
            out[m.name].append(cen)
            contrast[m.name].append(np.median(pk))
    sh = np.array(shake) if shake else None
    disp = {}
    for m in members:
        d = np.array(out[m.name])
        if sh is not None:
            d = d - (sh @ m.normal)[:, None]
        disp[m.name] = d
    quality = {}
    for m in members:
        d = disp[m.name]
        # second difference / sqrt(6) approximates white position noise at high frame rates
        hf = float(np.std(np.diff(d, 2, axis=0)) / np.sqrt(6)) if d.shape[0] > 3 else float("nan")
        quality[m.name] = {"contrast": float(np.median(contrast[m.name])), "noise_px": hf}
    stations = next(iter(grids.values()))[2]
    return Tracking(tb, members, stations, disp, quality, sh)
