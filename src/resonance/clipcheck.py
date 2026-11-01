"""Check a clip before measuring: is it the original, how fast is it really, did the camera move?

Runs in a few seconds on a low-resolution copy, so the person filming can re-shoot on the spot instead of
finding out after the analysis. Every threshold here matches a gate in `analyze`, so a clip that passes
the check is not refused later for a reason the check could have caught.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np

from .displacement import _hann2d, phase_shift
from .timing import frame_times
from .video import probe, read_frames

SHAKE_WARN_PX = 1.0
SHAKE_GATE_PX = 5.0   # the camera gate in analyze, measured there away from the marked members
SHAKE_FAIL_PX = 30.0  # the whole picture moving this much is the camera, whatever fills the frame


def _tags(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=bit_rate,duration:format_tags:stream_tags",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    tags = {k.lower(): v for k, v in info.get("format", {}).get("tags", {}).items()}
    for s in info.get("streams", []):
        tags.update({k.lower(): v for k, v in s.get("tags", {}).items()})
    return tags


def _camera(tags: dict) -> str | None:
    make = tags.get("com.apple.quicktime.make") or tags.get("make")
    model = tags.get("com.apple.quicktime.model") or tags.get("model")
    if tags.get("com.android.version") or tags.get("com.android.capture.fps"):
        make = make or "Android"
    return " ".join(x for x in (make, model) if x) or None


def _keep_mask(h: int, w: int, k: float, exclude) -> np.ndarray:
    """True where the camera may be judged: outside every excluded box (x0, y0, x1, y1) and every band
    ((x0, y0), (x1, y1), half_width) along a member's line. Coordinates in full-resolution pixels."""
    keep = np.ones((h, w), bool)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) * k
    for e in exclude or []:
        if len(e) == 4:
            x0, y0, x1, y1 = e
            keep &= ~((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1))
        else:
            (ax, ay), (bx, by), hw = e
            dx, dy = bx - ax, by - ay
            t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
            keep &= np.hypot(xs - (ax + t * dx), ys - (ay + t * dy)) > hw
    return keep


def global_shake(path: str, width: int = 320, max_frames: int = 3000, grid: tuple[int, int] = (4, 3),
                 exclude: list | None = None) -> np.ndarray | None:
    """Camera motion through the clip, in full-resolution pixels ([samples, 2]), or None if it cannot be judged.

    The frame is cut into tiles; each textured tile is matched against a keyframe by phase correlation and the
    median shift across tiles is taken, so a cable or string moving inside a few tiles does not count as camera
    motion while a moving camera shifts them all. When the tiles stop matching the keyframe (the view has
    changed), the current frame becomes the new keyframe and the motion so far is carried over. Long clips are
    sampled evenly (at most `max_frames` frames) so the whole clip is covered.
    `exclude`: boxes and member bands (see `_keep_mask`) left out of the estimate.
    None means fewer than three textured tiles were left, at the start or at a re-key: the caller must treat the
    camera as unchecked, never as still.
    """
    info = probe(path)
    full = info.width
    k = full / min(width, full)
    stride = max(1, -(-(info.frames or 0) // max_frames))
    nx, ny = grid
    pos, base, tiles, keep = [], np.zeros(2), None, None
    for i, img in enumerate(read_frames(path, width=width, gray=True)):
        if i % stride:
            continue
        if len(pos) >= max_frames:  # also bounds clips whose container gives no frame count
            break
        g = img.astype(np.float32)
        if tiles is None:  # (re)key: textured tiles of this frame, away from the members
            h, w = g.shape
            th, tw = h // ny, w // nx
            if keep is None:
                keep = _keep_mask(h, w, k, exclude)
            win = _hann2d(th, tw)
            tiles = []
            for r in range(ny):
                for c in range(nx):
                    sl = (slice(r * th, (r + 1) * th), slice(c * tw, (c + 1) * tw))
                    t = g[sl]
                    if keep[sl].mean() > 0.5 and t.std() >= 3.0:
                        tiles.append((sl, np.fft.fft2((t - t.mean()) * win)))
            if len(tiles) < 3:
                return None
            last = np.zeros(2)
        shifts, quality = [], []
        for sl, ref in tiles:
            q: list = []
            shifts.append(phase_shift(ref, g[sl], win, q))
            quality.append(q[0])
        good = [sh for sh, q in zip(shifts, quality) if q >= 0.15]
        if len(good) < max(3, len(tiles) // 3):  # the view no longer matches: carry the motion over, re-key
            base = base + last
            tiles = None
            pos.append(base.copy())
            continue
        last = np.median(np.array(good), axis=0)
        pos.append(base + last)
    return np.array(pos) * k


def check(path: str, capture_fps: float | None = None) -> dict:
    info = probe(path)
    tb = frame_times(path, capture_fps=capture_fps)
    tags = _tags(path)
    duration = float(tb.t[-1])
    fs = tb.fps
    items = []

    def add(key, status, value, text):
        items.append({"key": key, "status": status, "value": value, "text": text})

    cam = _camera(tags)
    add("source", "ok" if cam else "warn", cam,
        f"original file from {cam}" if cam else
        "no camera metadata: if this was sent through a chat app it was probably re-encoded at normal speed; "
        "copy the original file instead")

    f_hi = fs / 2.5
    if tb.ramped:
        add("frame rate", "fail", round(fs, 1), "the frame interval changes inside the clip (exported slow motion "
            "with real-time ends): use the original file, or trim to the slowed part")
    else:
        add("frame rate", "ok" if fs >= 200 else "warn", round(fs, 1),
            f"{fs:.0f} frames per second of real time ({tb.source}); frequencies up to {f_hi:.0f} Hz can be "
            "checked without aliasing" + ("" if fs >= 200 else "; use slow motion (240 fps) for short cables"))

    add("timing", "ok" if not tb.variable else "warn", round(tb.jitter, 4),
        "frame times are regular" if not tb.variable else
        f"frame times vary by {tb.jitter:.1%}; the spectrum uses the real times, but expect wider uncertainty")

    add("duration", "ok" if duration >= 4 else "warn" if duration >= 2 else "fail", round(duration, 2),
        f"{duration:.1f} s gives {1 / duration:.2f} Hz resolution" +
        ("" if duration >= 4 else "; film at least 4 s of steady vibration"))

    sh = global_shake(path)
    if sh is None or len(sh) < 2:
        rms = None
        add("camera", "warn", None, "too little static texture in view to check the camera: mark a static, "
            "textured anchor (a tower, a wall) so camera motion can be removed")
    else:
        rms = float(np.sqrt(np.mean(np.sum((sh - sh.mean(0)) ** 2, 1))))
    if rms is not None:
        add("camera", "ok" if rms <= SHAKE_WARN_PX else "warn" if rms <= SHAKE_FAIL_PX else "fail", round(rms, 2),
        f"camera steady ({rms:.1f} px RMS)" if rms <= SHAKE_WARN_PX else
        f"picture moved {rms:.1f} px RMS: mark a static textured patch as the anchor" if rms <= SHAKE_GATE_PX else
        f"picture moved {rms:.0f} px RMS: if the cable fills the view this is the cable itself, and the "
        f"measurement re-checks the camera away from the cables you mark (refused above {SHAKE_GATE_PX:.0f} px)"
        if rms <= SHAKE_FAIL_PX else
        f"picture moved {rms:.0f} px RMS: the camera itself moved; brace the phone (tripod, railing) and film again")

    first = next(iter(read_frames(path, width=640, gray=True))).astype(np.float32)
    clipped, mean = float(np.mean(first >= 250)), float(first.mean())
    add("exposure", "ok" if clipped < 0.05 and mean >= 40 else "warn", round(mean, 1),
        "exposure fine" if clipped < 0.05 and mean >= 40 else
        f"{clipped:.0%} of the frame is blown out: lock exposure on the cable (tap and hold)" if clipped >= 0.05 else
        "dark clip: slow motion needs daylight, the tracking will be noisy")

    worst = "fail" if any(i["status"] == "fail" for i in items) else \
        "warn" if any(i["status"] == "warn" for i in items) else "ok"
    f_lo = 3.0 / duration
    return {"video": path, "width": info.width, "height": info.height, "frames": int(tb.t.size),
            "fps": fs, "duration_s": duration, "status": worst, "checks": items,
            "band_hz": [round(f_lo, 2), round(f_hi, 1)],
            "summary": (f"Usable for cables whose fundamental lies between {f_lo:.2f} and {f_hi:.0f} Hz; "
                        f"a verified harmonic ladder needs three harmonics, so f0 up to about {fs / 6:.0f} Hz.")
            if worst != "fail" else "Film again before measuring: " +
            "; ".join(i["text"] for i in items if i["status"] == "fail") + "."}
