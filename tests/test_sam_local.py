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


def test_duplicate_masks_of_one_cable_are_merged(monkeypatch):
    seg = _fake_segment()
    dup = seg["masks"][0].copy()
    dup[:, 221:223] = True                                            # same cable, slightly fatter mask
    seg = {"boxes": np.vstack([seg["boxes"], [[218, 40, 223, 360]]]),
           "scores": np.append(seg["scores"], 0.5).astype(np.float32), "masks": np.vstack([seg["masks"], dup[None]])}
    monkeypatch.setattr(sam_local, "segment", lambda frame, concept, threshold=0.3: seg)
    members, extra = sam_local.find_members(np.zeros((400, 300, 3), np.uint8), "rope")
    assert len(members) == 3 and extra[2]["score"] == pytest.approx(0.9)


def test_mask_with_handrail_and_stray_blob_keeps_the_hanger_line():
    from resonance.members import from_mask
    m = np.zeros((400, 300), bool)
    m[20:380, 147:154] = True          # the hanger, 7 px wide
    m[100:112, 150:290] = True         # a strip of handrail SAM included, touching the hanger
    m[300:340, 10:60] = True           # a separate blob
    mem = from_mask("h", m)
    assert abs(mem.p0[0] - 150) < 2 and abs(mem.p1[0] - 150) < 2
    assert mem.meta["width_px"] == pytest.approx(7, abs=1.5) and mem.length_px > 320


def test_worker_failure_is_reported(monkeypatch, tmp_path):
    bad = tmp_path / "py"
    bad.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    bad.chmod(0o755)
    monkeypatch.setenv("RESONANCE_SAM_PYTHON", str(bad))
    with pytest.raises(RuntimeError, match="boom"):
        sam_local.segment(np.zeros((8, 8, 3), np.uint8), "rope")
