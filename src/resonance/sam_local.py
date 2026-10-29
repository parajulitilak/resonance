"""SAM 3.1 on the laptop: no network, no API bill, and real masks instead of boxes.

The model runs in a separate Python environment that has mlx-vlm (Apple silicon only), named by
RESONANCE_SAM_PYTHON; RESONANCE_SAM_MODEL is a local folder or hub id (default mlx-community/sam3.1-bf16).
Each mask becomes a member through its principal axis (`members.from_mask`), and its outline is kept so
the page can draw what SAM actually found.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .members import Member, from_mask

WORKER = Path(__file__).with_name("sam_mlx_worker.py")
DEFAULT_MODEL = "mlx-community/sam3.1-bf16"


def available() -> bool:
    py = os.environ.get("RESONANCE_SAM_PYTHON")
    return bool(py) and Path(py).exists()


def segment(frame_rgb: np.ndarray, concept: str, threshold: float = 0.3, timeout: float = 600) -> dict:
    py = os.environ.get("RESONANCE_SAM_PYTHON") or sys.executable
    model = os.environ.get("RESONANCE_SAM_MODEL", DEFAULT_MODEL)
    with tempfile.TemporaryDirectory() as d:
        png, out = Path(d) / "frame.png", Path(d) / "out.npz"
        Image.fromarray(frame_rgb).save(png)
        r = subprocess.run([py, str(WORKER), str(png), concept, model, str(out), str(threshold)],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"SAM 3.1 (MLX) failed: {r.stderr.strip()[-600:]}")
        z = np.load(out)
        return {"boxes": z["boxes"], "scores": z["scores"], "masks": z["masks"]}


def outline(mask: np.ndarray, member: Member, stations: int = 40) -> list[list[float]]:
    """A polygon around an elongated mask: its width at stations along the member's axis."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    p0 = np.asarray(member.p0, float)
    u = member.axis
    n = np.array([-u[1], u[0]])
    a, b = (pts - p0) @ u, (pts - p0) @ n
    edges = np.linspace(a.min(), a.max(), stations + 1)
    left, right = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (a >= lo) & (a <= hi)
        if not sel.any():
            continue
        s = 0.5 * (lo + hi)
        left.append(p0 + s * u + b[sel].min() * n)
        right.append(p0 + s * u + b[sel].max() * n)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in left + right[::-1]]


def find_members(frame_rgb: np.ndarray, concept: str = "rope", prefix: str = "m", min_score: float = 0.3,
                 min_elongation: float = 8.0) -> tuple[list[Member], list[dict]]:
    """Members (left to right) and, for each, SAM's score and mask outline. Blobs (weights, people) are
    dropped: a cable's mask is at least `min_elongation` times longer than it is thick."""
    seg = segment(frame_rgb, concept, threshold=min_score)
    found = []
    for box, score, mask in zip(seg["boxes"], seg["scores"], seg["masks"]):
        try:
            m = from_mask("tmp", mask)
        except ValueError:
            continue
        thickness = max(m.half_width / 2.5, 1.0)
        if m.length_px / thickness < min_elongation:
            continue
        found.append((0.5 * (m.p0[0] + m.p1[0]), m, float(score), mask))
    found.sort(key=lambda t: t[0])
    members, extra = [], []
    for i, (_, m, score, mask) in enumerate(found):
        m.name = f"{prefix}{i + 1}"
        members.append(m)
        extra.append({"name": m.name, "score": score, "outline": outline(mask, m)})
    return members, extra
