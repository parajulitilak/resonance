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
    weights: dict[str, np.ndarray] | None = None  # name -> per-station weight (tracking contrast)

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


def _centroids(profile: np.ndarray, offsets: np.ndarray, polarity) -> tuple[np.ndarray, np.ndarray]:
    """Contrast-weighted centroid per station; also returns peak contrast per station."""
    j = profile.shape[1]
    edge = max(2, j // 5)
    bg = np.median(np.concatenate([profile[:, :edge], profile[:, -edge:]], 1), axis=1, keepdims=True)
    pol = np.asarray(polarity, float).reshape(-1, 1) if np.ndim(polarity) else polarity
    c = np.clip(pol * (profile - bg), 0, None)
    peak = c.max(axis=1, keepdims=True)
    c = np.clip(c - 0.25 * peak, 0, None)
    w = c.sum(1)
    return (c * offsets).sum(1) / np.maximum(w, 1e-9), peak[:, 0]


def track(path: str, members: list[Member], anchor: tuple[int, int, int, int] | None = None,
          n_stations: int = 32, capture_fps: float | None = None,
          max_frames: int | None = None) -> Tracking:
    """Track every member through the video. `anchor` is a static box (x0, y0, x1, y1)."""
    tb = frame_times(path, capture_fps)
    patches = [m for m in members if m.kind == "patch"]
    members_all = members
    members = [m for m in members if m.kind != "patch"]
    pref: dict[str, tuple] = {}
    pshift: dict[str, list] = {m.name: [] for m in patches}
    grids = {m.name: m.sample_grid(n_stations) for m in members}
    polarity: dict[str, np.ndarray] = {}
    warm: dict[str, list] = {m.name: [] for m in members}
    n_warm = 12
    out: dict[str, list] = {m.name: [] for m in members}
    contrast: dict[str, list] = {m.name: [] for m in members}
    shake, ref_fft, win = [], None, None
    for i, frame in enumerate(read_frames(path, gray=True)):
        if max_frames and i >= max_frames:
            break
        f = frame.astype(np.float32)
        for m in patches:
            x0, y0, x1, y1 = m.box
            crop = f[y0:y1, x0:x1]
            if m.name not in pref:
                w_ = _hann2d(*crop.shape)
                pref[m.name] = (np.fft.fft2((crop - crop.mean()) * w_), w_)
            pshift[m.name].append(phase_shift(pref[m.name][0], crop, pref[m.name][1]))
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
                # per station: the sign of the strongest deviation from the profile edges, over the
                # first frames (robust to an off-axis start or a background that changes along the member)
                warm[m.name].append(prof)
                if len(warm[m.name]) < n_warm:
                    continue
                med = np.median(np.stack(warm[m.name]), 0)
                j = med.shape[1]
                bg = np.median(np.concatenate([med[:, :j // 5], med[:, -(j // 5):]], 1), axis=1, keepdims=True)
                dev = med - bg
                pol = np.sign(dev[np.arange(dev.shape[0]), np.argmax(np.abs(dev), 1)])
                pol[pol == 0] = -1.0
                polarity[m.name] = pol
                for wp in warm[m.name]:
                    cen, pk = _centroids(wp, o, polarity[m.name])
                    out[m.name].append(cen)
                    contrast[m.name].append(pk)
                continue
            cen, pk = _centroids(prof, o, polarity[m.name])
            out[m.name].append(cen)
            contrast[m.name].append(pk)
    sh = np.array(shake) if shake else None
    disp = {}
    for m in members:
        d = np.array(out[m.name])
        if sh is not None:
            d = d - (sh @ m.normal)[:, None]
        disp[m.name] = d
    quality, weights = {}, {}
    for m in patches:
        d2 = np.array(pshift[m.name])
        if sh is not None:
            d2 = d2 - sh[: d2.shape[0]]
        disp[m.name] = (d2 @ m.normal)[:, None]
        weights[m.name] = np.ones(1)
        quality[m.name] = {"contrast": float("nan"), "hf_motion_px": float(np.std(np.diff(disp[m.name][:, 0], 2)) / np.sqrt(6)),
                           "stations_used": 1, "stations": 1, "kind": "patch"}
    for m in members:
        d = disp[m.name]
        cs = np.median(np.array(contrast[m.name]), 0)        # per-station contrast
        w = np.clip(cs / max(np.median(cs), 1e-6), 0, 1) ** 2
        w[cs < 0.3 * np.median(cs)] = 0.0                  # stations that lost the member
        weights[m.name] = w
        # second difference / sqrt(6): a high-frequency motion proxy (includes signal), not pure noise
        hf = float(np.std(np.diff(d, 2, axis=0)) / np.sqrt(6)) if d.shape[0] > 3 else float("nan")
        quality[m.name] = {"contrast": float(np.median(cs)), "hf_motion_px": hf,
                           "stations_used": int((w > 0).sum()), "stations": int(w.size)}
    stations = next(iter(grids.values()))[2] if grids else np.array([0.5])
    return Tracking(tb, members_all, stations, disp, quality, sh, weights)
