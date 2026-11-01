from resonance.takes import confirm


def _rep(fps, f0, status="provisional", n_modes=4, reasons=None):
    return {"timing": {"fps": fps}, "video": f"{fps}.mp4", "members": [{
        "name": "r1", "ladder": {"status": status, "ok": status == "verified", "f0_hz": f0, "f0_sigma_hz": 0.01,
                                 "b": 0.0, "modes": [{"n": n} for n in range(1, n_modes + 1)],
                                 "reasons": reasons or ["no frequency prior: aliasing can't be ruled out"]},
        "tension": {"value_n": 1.0, "inputs": {"length_m": 1.8, "mass_per_m": 0.022}}}]}


def test_second_take_at_other_rate_verifies():
    out = confirm(_rep(240, 12.13), _rep(120, 12.14, n_modes=2))
    assert out["members"][0]["ladder"]["status"] == "verified"
    assert out["members"][0]["tension"]["value_n"] > 10


def test_disagreeing_takes_refused():
    out = confirm(_rep(240, 12.13), _rep(120, 3.9, n_modes=3))
    assert out["members"][0]["ladder"]["status"] == "refused"


def test_same_rate_is_not_an_alias_check():
    out = confirm(_rep(240, 12.13), _rep(240, 12.13))
    assert out["members"][0]["ladder"]["status"] == "provisional"


def test_approval_is_one_time_and_hash_bound(tmp_path):
    import json
    import pytest
    from resonance.agent import approve
    (tmp_path / "report.json").write_text(json.dumps({"timing": {"frames": 1, "fps": 240.0, "source": "t"},
                                                      "members": []}))
    report = {"summary": "s", "members": [{"name": "r1", "status": "refused", "statement": "Refused."}], "next_steps": []}
    verdicts = [{"id": 0, "kind": "statement", "member": "r1", "verdict": "agree", "reason": "ok"},
                {"id": 1, "kind": "summary", "member": "summary", "verdict": "agree", "reason": "ok"}]
    (tmp_path / "agent.json").write_text(json.dumps({"actions": {"flags": [], "reshoots": [], "report": report},
                                                     "violations": [], "verdicts": verdicts, "mode": "bench"}))
    approve(tmp_path, "A")
    rec = json.loads((tmp_path / "approval.json").read_text())
    assert rec["by"] == "A" and len(rec["sha256"]) == 64
    with pytest.raises(RuntimeError):
        approve(tmp_path, "B")


def test_approval_blocked_when_verification_failed_or_incomplete(tmp_path):
    import json
    import pytest
    from resonance.agent import approve
    (tmp_path / "report.json").write_text(json.dumps({"timing": {"frames": 1, "fps": 240.0, "source": "t"},
                                                      "members": []}))
    report = {"summary": "s", "members": [{"name": "r1", "status": "measured", "statement": "T 25.0 N."}], "next_steps": []}
    for error, verdicts in [("ConnectionResetError: reset", []), (None, [])]:
        (tmp_path / "agent.json").write_text(json.dumps({"actions": {"flags": [], "reshoots": [], "report": report},
                                                         "violations": [], "verdicts": verdicts, "mode": "bench",
                                                         "error": error}))
        with pytest.raises(RuntimeError):
            approve(tmp_path, "A")


def test_missing_verdict_never_renders_as_agrees():
    from resonance.agent import render_markdown
    rep = {"timing": {"frames": 1, "fps": 240.0, "source": "t"}, "members": [{"name": "r1", "ladder": {}}]}
    acts = {"report": {"summary": "s", "members": [{"name": "r1", "status": "measured", "statement": "T 25.0 N."}],
                       "next_steps": []}, "flags": [], "reshoots": []}
    md = render_markdown(rep, acts, [], [], "bench")
    assert "| needs engineer |" in md and "agrees" not in md.split("## Independent")[0]


def test_unanswered_disagreements_do_not_allow_signing(tmp_path):
    import json
    import pytest
    from resonance.agent import approve
    (tmp_path / "report.json").write_text(json.dumps({"timing": {"frames": 1, "fps": 240.0, "source": "t"},
                                                      "members": []}))
    report = {"summary": "s", "members": [{"name": "r1", "status": "measured", "statement": "T 25.0 N."}], "next_steps": []}
    unanswered = [{"id": 0, "kind": "statement", "member": "r1", "verdict": "disagree", "answered": False,
                   "reason": "model call limit reached"},
                  {"id": 1, "kind": "summary", "member": "summary", "verdict": "disagree", "answered": False,
                   "reason": "model call limit reached"}]
    (tmp_path / "agent.json").write_text(json.dumps({"actions": {"flags": [], "reshoots": [], "report": report},
                                                     "violations": [], "verdicts": unanswered, "mode": "bench"}))
    with pytest.raises(RuntimeError, match="not every claim"):
        approve(tmp_path, "A")


def test_rule9_caveat_not_skipped_by_the_word_uncertainty():
    from resonance.agent import enforce_rules
    rep = {"members": [{"name": "m1", "ladder": {"status": "verified", "ok": True},
                        "visual_review": {"verdict": "rejected", "what": "floor lead"}}]}
    acts = {"report": {"summary": "s", "members": [{"name": "m1", "status": "measured",
                                                     "statement": "Tension 19.6 N with an uncertainty of 0.4 N."}],
                       "next_steps": []}, "flags": [], "reshoots": []}
    v = enforce_rules(rep, acts, "bench")
    assert "Identification of this member is uncertain" in acts["report"]["members"][0]["statement"]
    assert any("uncertain identification" in x["rule"] for x in v)
