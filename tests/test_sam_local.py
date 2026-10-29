import numpy as np
import pytest

from resonance import sam_local


def _fake_segment(h=400, w=300):
    masks, boxes = [], []
    for x, tilt in ((220, 0.0), (60, 0.05), (140, -0.03)):          # three thin cables, out of order
        m = np.zeros((h, w), bool)
        for y in range(40, 360):
            c = int(round(x + tilt * (y - 40)))
            m[y, c - 1:c + 2] = True
        masks.append(m)
        boxes.append([x - 2, 40, x + 2, 360])
    blob = np.zeros((h, w), bool)
    blob[300:360, 20:70] = True                                       # a hanging weight: not a cable
    masks.append(blob)
    boxes.append([20, 300, 70, 360])
    return {"boxes": np.array(boxes, np.float32), "scores": np.array([0.9, 0.8, 0.7, 0.95], np.float32),
            "masks": np.array(masks)}


def test_masks_become_members_left_to_right_and_blobs_are_dropped(monkeypatch):
    monkeypatch.setattr(sam_local, "segment", lambda frame, concept, threshold=0.3: _fake_segment())
    members, extra = sam_local.find_members(np.zeros((400, 300, 3), np.uint8), "rope")
    assert [m.name for m in members] == ["m1", "m2", "m3"]
    assert [round((m.p0[0] + m.p1[0]) / 2) for m in members] == pytest.approx([68, 135, 220], abs=3)
    assert all(m.length_px > 280 for m in members)
    assert len(extra[0]["outline"]) > 20 and extra[1]["score"] == pytest.approx(0.7)


def test_worker_failure_is_reported(monkeypatch, tmp_path):
    bad = tmp_path / "py"
    bad.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    bad.chmod(0o755)
    monkeypatch.setenv("RESONANCE_SAM_PYTHON", str(bad))
    with pytest.raises(RuntimeError, match="boom"):
        sam_local.segment(np.zeros((8, 8, 3), np.uint8), "rope")
