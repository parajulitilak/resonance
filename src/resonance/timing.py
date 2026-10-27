"""True frame timing from the container, not the nominal frame rate.

Slow-motion phone files are the trap: iPhone originals carry real 240 fps
timestamps but the actual rate can drift (users report 177-219 fps), and
Android slow-motion files are stamped at playback speed with the capture rate
stored in the `com.android.capture.fps` tag. Frequencies are only as good as
the time axis, so every spectrum in Resonance is computed from these times.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass
class TimeBase:
    t: np.ndarray            # seconds of real (capture) time, starting at 0, one per frame
    fps: float               # median capture rate
    nominal_fps: float       # what the container claims
    jitter: float            # std of frame interval / median interval (after de-quantising)
    time_scale: float        # factor applied to container time to get capture time
    source: str
    ramped: bool = False     # the frame interval changes during the clip (e.g. exported slow motion)
    dequantised: bool = False

    @property
    def variable(self) -> bool:
        return self.jitter > 0.02


def _probe_json(path: str, args: list[str]) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def capture_fps_tag(path: str) -> float | None:
    """Android slow motion: real capture rate stored as a format/stream tag."""
    info = _probe_json(path, ["-show_entries", "format_tags:stream_tags"])
    tags = dict(info.get("format", {}).get("tags", {}))
    for s in info.get("streams", []):
        tags.update(s.get("tags", {}))
    for k, v in tags.items():
        if k.lower() == "com.android.capture.fps":
            try:
                return float(v)
            except ValueError:
                return None
    return None


def _decoded_frame_times(path: str) -> np.ndarray:
    """Timestamps of the frames the decoder actually outputs (edit lists and discarded packets respected)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=best_effort_timestamp_time", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    vals = [float(v.split(",")[0]) for v in out.split() if v and v.split(",")[0] not in ("N/A", "")]
    return np.array(vals)


def frame_times(path: str, capture_fps: float | None = None) -> TimeBase:
    """Per-frame capture times, aligned with the frames `video.read_frames` yields.

    `capture_fps` overrides the detected capture rate (use it when a slow-motion file was exported at
    playback speed and its timestamps are uniform).
    """
    info = _probe_json(path, ["-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate"])
    num, den = info["streams"][0]["avg_frame_rate"].split("/")
    nominal = float(num) / float(den) if float(den) else float("nan")
    pts = _decoded_frame_times(path)
    if pts.size < 2:
        raise ValueError(f"no frame timestamps in {path}")
    t = pts - pts[0]
    n = t.size
    idx = np.arange(n)

    # Ramps: a slowed middle with real-time ends shows up as a step in the local frame interval.
    dt = np.diff(t)
    k = min(31, max(5, (n // 10) | 1))
    local = np.array([np.median(dt[max(0, i - k // 2): i + k // 2 + 1]) for i in range(0, dt.size, max(1, k // 2))])
    ramped = bool(local.size > 2 and local.max() / max(local.min(), 1e-9) > 1.3)

    # Coarse timescales (QuickTime 1/600 at 240 fps) alternate 2- and 3-tick intervals: fit a line
    # through time vs frame index when the scatter is only quantisation and there are no real gaps.
    dequant = False
    if not ramped:
        a, c = np.polyfit(idx, t, 1)
        resid = t - (a * idx + c)
        if np.std(resid) < 0.35 * a and dt.max() < 1.6 * a:
            t, dequant = a * idx + c - c, True

    container_fps = (n - 1) / (t[-1] - t[0])
    scale, source = 1.0, "decoded frame timestamps"
    tag = capture_fps_tag(path)
    if capture_fps:
        scale, source = container_fps / capture_fps, f"override capture_fps={capture_fps:g}"
    elif tag and tag > container_fps * 1.5:
        scale, source = container_fps / tag, f"android capture tag {tag:g} fps"
    t = t * scale
    dt = np.diff(t)
    med = float(np.median(dt))
    return TimeBase(t=t, fps=(n - 1) / (t[-1] - t[0]), nominal_fps=nominal, jitter=float(np.std(dt) / med),
                    time_scale=scale, source=source + (", de-quantised" if dequant else ""),
                    ramped=ramped, dequantised=dequant)
