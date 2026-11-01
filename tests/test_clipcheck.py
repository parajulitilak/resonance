import subprocess

import pytest

from resonance.clipcheck import check


def _clip(path, fps, seconds, vf=None, size="320x240"):
    """A still textured frame held for `seconds` (testsrc2 alone animates part of its pattern)."""
    n = int(fps * seconds)
    chain = f"trim=end_frame=1,loop=loop={n - 1}:size=1:start=0,setpts=N/{fps}/TB" + (f",{vf}" if vf else "")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}", "-vf", chain,
                    "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def _by_key(r):
    return {c["key"]: c for c in r["checks"]}


def test_steady_slow_motion_clip_passes_with_its_band(tmp_path):
    r = check(_clip(tmp_path / "s.mp4", 240, 5))
    c = _by_key(r)
    assert c["frame rate"]["status"] == "ok" and c["camera"]["status"] == "ok"
    assert r["fps"] == pytest.approx(240, rel=2e-3)
    assert r["band_hz"][1] == pytest.approx(96, rel=1e-2)
    assert r["status"] != "fail"


def test_shaky_clip_fails_on_camera_before_any_analysis(tmp_path):
    vf = "crop=960:540:160+90*sin(2*PI*1.5*t):90+60*sin(2*PI*1.1*t)"   # a hand-held camera, about 75 px RMS
    r = check(_clip(tmp_path / "h.mp4", 60, 3, vf, size="1280x720"))
    c = _by_key(r)
    assert c["camera"]["status"] == "fail" and c["camera"]["value"] > 30
    assert c["frame rate"]["status"] == "warn"
    assert r["status"] == "fail" and r["summary"].startswith("Film again")


def test_moderate_picture_motion_is_left_to_the_measurement(tmp_path):
    # a cable filling a close-up moves the whole picture too; the analysis re-checks away from the members
    vf = "crop=256:192:32+20*sin(2*PI*1.5*t):24+10*sin(2*PI*1.1*t)"
    c = _by_key(check(_clip(tmp_path / "m.mp4", 60, 3, vf)))
    assert c["camera"]["status"] == "warn" and 5 < c["camera"]["value"] < 30


def test_short_clip_is_flagged(tmp_path):
    r = check(_clip(tmp_path / "t.mp4", 240, 1.5))
    assert _by_key(r)["duration"]["status"] == "fail"


def test_band_exclusion_keeps_the_frame_around_a_diagonal_cable():
    import numpy as np

    from resonance.clipcheck import _keep_mask
    keep = _keep_mask(180, 320, 6.0, [((0, 0), (1920, 1080), 40.0)])
    assert keep.mean() > 0.85                    # a bounding box would have removed everything
    assert not keep[90, 160]                     # the cable's own pixels are left out


def test_moving_camera_with_a_diagonal_cable_is_still_measured(tmp_path):
    from resonance.analyze import picture_motion
    from resonance.members import Member
    vf = "crop=960:540:160+90*sin(2*PI*1.5*t):90+60*sin(2*PI*1.1*t)"
    v = _clip(tmp_path / "d.mp4", 60, 3, vf, size="1280x720")
    assert picture_motion(v, [Member("diag", (0, 0), (960, 540), half_width=10)]) > 30
    rows = [Member(f"h{i}", (60 + 100 * i, 0), (60 + 100 * i, 540), half_width=8) for i in range(9)]
    assert picture_motion(v, rows) > 30          # a row of hangers no longer hides the camera


def test_unjudgeable_camera_refuses_instead_of_passing(tmp_path):
    import math

    import numpy as np

    from resonance.analyze import analyze_tracking, picture_motion
    from resonance.displacement import Tracking
    from resonance.members import Member
    from resonance.timing import TimeBase
    v = _clip(tmp_path / "s.mp4", 60, 2)
    assert math.isnan(picture_motion(v, [Member("all", (0, 120), (320, 120), half_width=200)]))
    fs = 240.0
    t = np.arange(1200) / fs
    s = np.linspace(0.06, 0.94, 32)
    disp = sum(np.sin(np.pi * n * s)[None, :] * np.sin(2 * np.pi * 8.0 * n * t)[:, None] / n for n in (1, 2, 3, 4))
    tb = TimeBase(t=t, fps=fs, nominal_fps=fs, jitter=0.0, time_scale=1.0, source="test")
    m = Member("rope", (100, 50), (100, 900), f0_range=(5.0, 12.0))
    rep = analyze_tracking(Tracking(tb, [m], s, {"rope": disp}, {"rope": {}}, None, {"rope": np.ones(32)}),
                           picture_motion_px=float("nan"))
    lad = rep["members"][0]["ladder"]
    assert lad["status"] == "refused" and "could not be checked" in lad["reasons"][-1]
