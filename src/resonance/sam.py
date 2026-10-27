"""SAM 3.1 on Meta Model API: find every cable in a frame and turn each into a member.

One concept prompt ("rope", "suspension cable", "hanger rod") returns every
instance with a box and a mask, so one request places N sensors. Members are
built from the boxes (robust, no mask decoding needed): a thin member's box
spans its length; for an inclined member we pick the box diagonal along which
the image actually shows a line. Wire format: https://dev.meta.ai/docs/sam/reading-segmentation
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from .llm import load_env
from .members import Member

API = "https://api.meta.ai/v1/responses"

BOX = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>")
RECORD = re.compile(r"(?:^|,)(\d+)(<\|box;[^|]*\|>)(<\|mask;x=0;y=0(?:;c=[\d.]+)?;data=(\d+),(\d+),(.)(.*?)\|>)"
                    r"(?=,\d+<\|box|$)", re.S)
FRAME = re.compile(r"^<(\d+)f>")


@dataclass
class Detection:
    frame: int
    obj: int
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 inclusive, media pixels
    size: tuple[int, int]            # w, h of the media
    conf: float | None
    mask_hw: tuple[int, int]
    mask_enc: str
    mask_payload: str


def parse_line(line: str) -> list[Detection]:
    m = FRAME.match(line.strip())
    if not m:
        return []
    frame, body = int(m.group(1)), line.strip()[m.end():]
    out = []
    for r in RECORD.finditer(body):
        b = BOX.match(r.group(2))
        x1, y1, x2, y2, w, h = (int(v) for v in b.groups()[:6])
        conf = float(b.group(7)) if b.group(7) else None
        out.append(Detection(frame, int(r.group(1)), (x1, y1, x2, y2), (w, h), conf,
                             (int(r.group(4)), int(r.group(5))), r.group(6), r.group(7)))
    return out


def parse_output(text: str) -> list[Detection]:
    return [d for line in text.splitlines() for d in parse_line(line)]


def _data_url(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def segment_image(img: np.ndarray, concept: str, timeout: int = 120) -> list[Detection]:
    """Concept-prompt SAM 3.1 on one RGB frame. Streams and returns all detections."""
    load_env()
    key = os.environ.get("MODEL_API_KEY", "")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set")
    body = {"model": "sam-3.1", "stream": True, "metadata": {"include_confidence": "true"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_image", "image_url": _data_url(img)}]}]}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "text/event-stream"})
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "response.output_text.delta":
                text.append(ev.get("delta", ""))
    return parse_output("\n".join(text))  # one delta per frame line


def _line_contrast(gray: np.ndarray, p0, p1, n: int = 64) -> float:
    s = np.linspace(0, 1, n)
    xs, ys = p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1])
    on = map_coordinates(gray, [ys, xs], order=1, mode="nearest")
    d = np.array([p1[0] - p0[0], p1[1] - p0[1]], float)
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * 6
    off = 0.5 * (map_coordinates(gray, [ys + nrm[1], xs + nrm[0]], order=1, mode="nearest")
                 + map_coordinates(gray, [ys - nrm[1], xs - nrm[0]], order=1, mode="nearest"))
    return float(np.median(np.abs(on - off)))


def member_from_box(name: str, det: Detection, gray: np.ndarray, **kw) -> Member:
    """Thin box → its long axis; wide box → whichever diagonal shows a line."""
    x1, y1, x2, y2 = det.box
    w, h = x2 - x1 + 1, y2 - y1 + 1
    if w <= 0.15 * h:   # near-vertical
        cx = 0.5 * (x1 + x2)
        return Member(name, (cx, y1), (cx, y2), half_width=max(6.0, 1.5 * w), **kw)
    if h <= 0.15 * w:   # near-horizontal
        cy = 0.5 * (y1 + y2)
        return Member(name, (x1, cy), (x2, cy), half_width=max(6.0, 1.5 * h), **kw)
    a = _line_contrast(gray, (x1, y1), (x2, y2))
    b = _line_contrast(gray, (x1, y2), (x2, y1))
    p0, p1 = ((x1, y1), (x2, y2)) if a >= b else ((x1, y2), (x2, y1))
    return Member(name, p0, p1, half_width=8.0, **kw)


def find_members(frame_rgb: np.ndarray, concept: str = "rope", prefix: str = "m",
                 min_conf: float = 0.3) -> list[Member]:
    dets = [d for d in segment_image(frame_rgb, concept) if d.conf is None or d.conf >= min_conf]
    dets.sort(key=lambda d: (d.box[0] + d.box[2]) / 2)  # left to right
    gray = frame_rgb.astype(np.float32).mean(2)
    return [member_from_box(f"{prefix}{i + 1}", d, gray) for i, d in enumerate(dets)]
