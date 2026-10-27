"""One test per failure reproduced in the 2026-10-28 independent code review."""
import json
import math
import subprocess

import numpy as np
import pytest

from resonance.agent import enforce_rules, render_markdown
from resonance.analyze import analyze_tracking
from resonance.displacement import Tracking
from resonance.members import Member
from resonance.physics import tension
from resonance.sam import parse_line
from resonance.spectrum import Peak, fit_harmonics
from resonance.timing import TimeBase, frame_times


def folded_peaks(f0, fs, n_max=12, snr=40.0):
    out = {}
    for n in range(1, n_max + 1):
        f = (n * f0) % fs
        f = min(f, fs - f)
        if f > 0.2:
            out.setdefault(round(f, 6), Peak(f, 1.0 / n ** 2, snr))
    return sorted(out.values(), key=lambda p: p.f)


@pytest.mark.parametrize("true_f0", [28.0, 31.0, 32.5, 58.0])
def test_critical1_aliased_ladder_never_verified(true_f0):
    pk = folded_peaks(true_f0, fs=30.0)
    assert not fit_harmonics(pk, fs=30.0).ok                       # no prior: at best provisional
    assert not fit_harmonics(pk, fs=30.0, f0_range=(5, 60)).ok     # prior beyond fs/2.5: refused


def test_critical2_dominant_third_mode_refused_by_mode_shape():
    fs, dur, f_true = 240.0, 4.0, 8.0
    t = np.arange(int(fs * dur)) / fs
    s = np.linspace(0.06, 0.94, 32)
    x = 3.0 * np.sin(3 * math.pi * s)[None, :] * np.sin(2 * math.pi * 3 * f_true * t)[:, None]
    x = x + 0.06 * x ** 2 + 0.01 * x ** 3                              # tracking nonlinearity → 2x, 3x terms
    x += np.random.default_rng(0).normal(0, 0.02, x.shape)
    tb = TimeBase(t=t, fps=fs, nominal_fps=fs, jitter=0.0, time_scale=1.0, source="test")
    m = Member("h1", (100, 100), (100, 900), length_m=2.0, mass_per_m=0.5, f0_range=(5, 40))
    tr = Tracking(tb, [m], s, {"h1": x}, {"h1": {"contrast": 100.0}}, None, None)
    rep = analyze_tracking(tr)
    lad = rep["members"][0]["ladder"]
    assert lad["status"] != "verified", (lad["f0_hz"], rep["members"][0]["mode_nodes"])


def test_critical3_wrong_time_base_caught_by_prior():
    true = [Peak(n * 12.0, 1 / n ** 2, 40) for n in range(1, 6)]
    slowed = [Peak(p.f / 8, p.power, p.snr) for p in true]           # exported slo-mo: 1/8 frequencies
    assert not fit_harmonics(slowed, fs=30.0, f0_range=(5, 40)).ok


def test_major4_stiff_member_assigned_and_verified():
    b = 4e-3
    pk = [Peak(n * 10.0 * math.sqrt(1 + b * n * n), 1 / n ** 2, 40) for n in range(1, 9)]
    fit = fit_harmonics(pk, fs=480.0, f0_range=(3, 60))
    assert fit.ok, fit.reasons
    assert fit.f0 == pytest.approx(10.0, rel=2e-3) and fit.b == pytest.approx(b, rel=0.05)


def test_major5_clamped_bracket_reported_for_moderate_stiffness():
    est = tension(10.0, 0.01, b=(math.pi / 20) ** 2, length_m=2.0, mass_per_m=0.5)   # xi = 20
    assert not est.reliable and est.clamped_n < 0.85 * est.tension_n


def test_major6_ratio_205_is_not_provisional():
    fit = fit_harmonics([Peak(10.0, 1, 40), Peak(20.5, 0.25, 40)], fs=240.0, f0_range=(3, 60))
    assert not fit.provisional and not fit.ok


def test_major6_clean_pair_is_provisional_with_honest_sigma():
    fit = fit_harmonics([Peak(10.0, 1, 40), Peak(20.02, 0.25, 40)], fs=240.0, f0_range=(3, 60), duration=5.0)
    assert fit.provisional and fit.b == 0.0 and fit.f0_sigma >= 0.01


def _rep(status="refused"):
    lad = {"ok": status == "verified", "status": status, "reasons": ["only 2 harmonics on the ladder (need 3)"],
           "f0_hz": 10.0, "f0_sigma_hz": 0.05, "modes": []}
    return {"timing": {"frames": 10, "fps": 240.0, "source": "t"},
            "members": [{"name": "H3", "ladder": lad, "tension": {"value_n": 120.0, "sigma_n": 3.0}, "peaks": [],
                         "quality": {}}]}


def test_major9_rules_catch_bypasses_bench():
    rep = _rep("refused")
    actions = {"flags": [], "reshoots": [], "report": {"summary": "ok", "next_steps": ["Hanger is safe to use."],
               "members": [{"name": "h3", "status": "refused", "statement": "f0 = 12.3 Hz so T = 380 N"}]}}
    v = enforce_rules(rep, actions, "bench")
    rules = " ".join(x["rule"] for x in v)
    assert "refused member" in rules and "banned wording" in rules
    assert "380" not in actions["report"]["members"][0]["statement"]


def test_major9_negated_provisional_rewritten():
    rep = _rep("provisional")
    actions = {"flags": [], "reshoots": [], "report": {"summary": "", "next_steps": [],
               "members": [{"name": "H3", "status": "measured", "statement": "Not provisional: T = 120 N"}]}}
    v = enforce_rules(rep, actions, "bench")
    assert v and actions["report"]["members"][0]["statement"].startswith("Provisional")


def test_major9_field_mode_kilonewtons_caught():
    rep = _rep("verified")
    actions = {"flags": [], "reshoots": [], "report": {"summary": "Main cable carries 41.2 kilonewtons.",
               "next_steps": [], "members": [{"name": "H3", "status": "measured", "statement": "41.2 kilonewtons"}]}}
    v = enforce_rules(rep, actions, "field")
    assert sum("field mode" in x["rule"] for x in v) >= 2


def test_major9_missing_verdict_renders_needs_engineer():
    rep = _rep("verified")
    actions = {"flags": [], "reshoots": [], "report": {"summary": "", "next_steps": [],
               "members": [{"name": "H3", "status": "measured", "statement": "f0 10.0 Hz"}]}}
    verdicts = [{"id": 0, "member": "H3", "claim": "x", "verdict": "disagree", "reason": "no verdict"}]
    assert "needs engineer" in render_markdown(rep, actions, [], verdicts, "bench")


def test_minor_sam_parser_tolerates_float_boxes_and_extra_keys():
    line = ("<0f>2<|box;x1=10.5;y1=20;x2=14;y2=400;w=720;h=1280;c=9e-1;r=0|>"
            "<|mask;x=0;y=0;data=10,10,~abc|>")
    d = parse_line(line)
    assert len(d) == 1 and d[0].box == (10, 20, 14, 400) and d[0].conf == pytest.approx(0.9)


def test_major10_quantised_timestamps_dequantised(tmp_path):
    vid = tmp_path / "q.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=240:duration=1",
                    "-c:v", "libx264", "-video_track_timescale", "600", str(vid)], check=True)
    tb = frame_times(str(vid))
    assert tb.fps == pytest.approx(240.0, rel=2e-3)
    assert tb.jitter < 0.02
