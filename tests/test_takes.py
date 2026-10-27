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
    (tmp_path / "agent.json").write_text(json.dumps({"actions": {"flags": [], "reshoots": [], "report": None},
                                                     "violations": [], "verdicts": [], "mode": "bench"}))
    approve(tmp_path, "A")
    rec = json.loads((tmp_path / "approval.json").read_text())
    assert rec["by"] == "A" and len(rec["sha256"]) == 64
    with pytest.raises(RuntimeError):
        approve(tmp_path, "B")
