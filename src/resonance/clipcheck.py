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

SHAKE_FAIL_PX = 5.0   # same as the camera gate in analyze
SHAKE_WARN_PX = 1.0


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


def global_shake(path: str, width: int = 256, max_frames: int = 3000) -> np.ndarray:
    """Whole-frame shift of every frame against the first, in full-resolution pixels ([frames, 2])."""
    full = probe(path).width
    k = full / min(width, full)
    shifts, ref, win = [], None, None
    for i, img in enumerate(read_frames(path, width=width, gray=True)):
        if i >= max_frames:
            break
        g = img.astype(np.float32)
        if ref is None:
            win = _hann2d(*g.shape)
            ref = np.fft.fft2((g - g.mean()) * win)
        shifts.append(phase_shift(ref, g, win))
    return np.array(shifts) * k


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
    rms = float(np.sqrt(np.mean(np.sum((sh - sh.mean(0)) ** 2, 1)))) if len(sh) > 1 else 0.0
    add("camera", "ok" if rms <= SHAKE_WARN_PX else "warn" if rms <= SHAKE_FAIL_PX else "fail", round(rms, 2),
        f"camera steady ({rms:.1f} px RMS)" if rms <= SHAKE_WARN_PX else
        f"picture moved {rms:.1f} px RMS: mark a static textured patch as the anchor" if rms <= SHAKE_FAIL_PX else
        f"picture moved {rms:.0f} px RMS: brace the phone (tripod, railing) and film again; if it is the "
        "structure that moves, mark a static anchor and the measurement will tell camera from structure")

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
