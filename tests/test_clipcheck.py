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
