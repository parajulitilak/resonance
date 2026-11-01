"""Visual review of each candidate member by Muse Spark Open, before anything is measured.

SAM 3.1 answers "where are things that look like a cable?", which also catches floor leads, frame bars and
handrails. Muse Spark Open looks at the frame with one candidate highlighted and answers a narrower
question: is this a tensioned member, and are both of its supports in view? A candidate it rejects is not
measured unless the person filming keeps it. If the review fails (network, unparseable answer) the member is
marked "unreviewed", never "confirmed".
"""

from __future__ import annotations

import base64
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from .llm import chat
from .members import Member

QUESTION = (
    "You are checking candidates for a vibration-based cable tension measurement. The magenta line marks ONE object "
    "that an image segmentation model proposed as a cable. Judge only the marked object.\n"
    "A tensioned member is a cable, rope, wire, string, hanger, stay or guy that is held in tension between two "
    "supports (it can vibrate like a string). Beams, frame bars, pipes resting on supports, handrails that are not "
    "cables, floor leads and shadows are not.\n"
    'Answer with JSON only: {"is_tensioned_member": true|false, "what_it_is": "<short noun phrase>", '
    '"both_supports_visible": true|false, "reason": "<one sentence>"}'
)


def _view(frame: np.ndarray, m: Member, width: int = 1024) -> str:
    im = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(im)
    lw = max(4, im.width // 300)
    d.line([tuple(m.p0), tuple(m.p1)], fill=(255, 0, 255), width=lw)
    for p in (m.p0, m.p1):
        d.ellipse([p[0] - 2 * lw, p[1] - 2 * lw, p[0] + 2 * lw, p[1] + 2 * lw], outline=(255, 0, 255), width=lw // 2 + 1)
    im.thumbnail((width, width))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(j.get("is_tensioned_member"), bool):
        return None
    return j


def review_member(frame: np.ndarray, m: Member, provider: str = "baseten", context: str = "") -> dict:
    q = QUESTION + (f"\nSite notes from the person filming (use them to recognise unusual members, but still judge "
                    f"what the marked object is): <notes>{context[:600]}</notes>" if context.strip() else "")
    try:
        r = chat([{"role": "user", "content": [{"type": "text", "text": q},
                                               {"type": "image_url", "image_url": {"url": _view(frame, m)}}]}],
                 provider=provider, max_tokens=3000, retries=2)
        j = _parse(r["message"].get("content"))
    except Exception as e:  # noqa: BLE001
        return {"name": m.name, "verdict": "unreviewed", "reason": f"review failed: {type(e).__name__}: {e}"[:300]}
    if j is None:
        return {"name": m.name, "verdict": "unreviewed", "reason": "the reviewer's answer could not be read"}
    verdict = "confirmed" if j["is_tensioned_member"] else "rejected"
    return {"name": m.name, "verdict": verdict, "what": str(j.get("what_it_is", ""))[:80],
            "both_supports_visible": j.get("both_supports_visible"), "reason": str(j.get("reason", ""))[:300],
            "model": r["model"]}


def review(frame: np.ndarray, members: list[Member], provider: str = "baseten", context: str = "",
           workers: int = 6) -> list[dict]:
    """One review per member, in the members' order (requests run in parallel)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda m: review_member(frame, m, provider, context), members))
