import json

import pytest

from resonance import autopilot as ap


def scripted(turns):
    """A fake model: each turn is one tool call (name, args) or a list of them; then it finishes."""
    seq = list(turns)

    def fake_chat(msgs, tools=None, provider="baseten", tool_choice=None, **kw):
        turn = seq.pop(0) if seq and not tool_choice else ("finish", {"summary": "done"})
        calls = turn if isinstance(turn, list) else [turn]
        tc = [{"id": f"c{len(msgs)}_{k}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
              for k, (n, a) in enumerate(calls)]
        return {"message": {"content": "", "tool_calls": tc}, "usage": {}, "model": "fake"}
    return fake_chat


def toolbox(notes, measured, flip=False):
    found = [{"p0": [10, 10], "p1": [10, 400], "half_width": 8, "score": 0.6},
             {"p0": [50, 300], "p1": [300, 310], "half_width": 8, "score": 0.5}]
    calls = {"n": 0}

    def review(cands):  # with flip=True a second review of the same candidate would say "confirmed"
        calls["n"] += 1
        return [{"verdict": "confirmed" if c["p0"] == [10, 10] or (flip and calls["n"] > 1) else "rejected", "what": "x"}
                for c in cands]
    return ap.Toolbox(
        check=lambda: {"status": "ok", "summary": "fine", "checks": []},
        find=lambda prompt: found if prompt == "suspension cable" else [],
        review=review,
        measure=lambda members: measured.append(members) or {"run": "r1", "members": [{"name": m["name"], "status": "provisional"} for m in members]},
        notes=notes)


def test_agent_cannot_measure_rejected_candidates_and_priors_are_built_by_code(monkeypatch):
    measured = []
    tb = toolbox("Footbridge hangers, design tension about 20 kN; first mode tuned near 5 Hz.", measured)
    monkeypatch.setattr(ap, "chat", scripted([
        ("check_clip", {}),
        ("find_cables", {"prompt": "cable"}),                       # nothing: the agent should try another prompt
        ("find_cables", {"prompt": "suspension cable"}),
        ("review_candidates", {"ids": ["c1", "c2"]}),
        ("measure", {"ids": ["c2"]}),                               # rejected by the review: blocked
        ("measure", {"ids": ["c1", "c2"], "f0_hz_in_notes": 5, "length_m_in_notes": 12.0}),
    ]))
    out = ap.run_autopilot(tb)
    blocked = next(s for s in out["steps"] if s["tool"] == "measure")
    assert "error" in blocked["result"] and blocked["result"]["not_confirmed"] == ["c2"]
    assert len(measured) == 1 and [m["name"] for m in measured[0]] == ["c1"]
    assert measured[0][0]["f0_range"] == [4.0, 6.0]                 # built by code from the noted 5 Hz
    assert "length_m" not in measured[0][0]                         # 12 m is not in the notes
    assert out["run"] == "r1" and out["summary"] == "done"


def test_a_wide_or_invented_prior_is_impossible(monkeypatch):
    measured = []
    tb = toolbox("first mode tuned near 5 Hz", measured)
    monkeypatch.setattr(ap, "chat", scripted([
        ("find_cables", {"prompt": "suspension cable"}), ("review_candidates", {"ids": ["c1"]}),
        ("measure", {"ids": ["c1"], "f0_hz_in_notes": 15, "f0_range_hz": [1, 20]}),
    ]))
    out = ap.run_autopilot(tb)
    m = [s for s in out["steps"] if s["tool"] == "measure"][0]["result"]
    assert "f0_range" not in measured[0][0] and "f0_hz_in_notes" in m["values_dropped"]


def test_first_verdict_is_final(monkeypatch):
    measured = []
    tb = toolbox("", measured, flip=True)
    monkeypatch.setattr(ap, "chat", scripted([
        ("find_cables", {"prompt": "suspension cable"}), ("review_candidates", {"ids": ["c2"]}),
        ("review_candidates", {"ids": ["c2"]}), ("review_candidates", {"ids": ["c2"]}),
        ("measure", {"ids": ["c2"]}),
    ]))
    out = ap.run_autopilot(tb)
    again = [s for s in out["steps"] if s["tool"] == "review_candidates"][1]["result"]["reviews"][0]
    assert again["verdict"] == "rejected" and "final" in again["note"] and not measured


def test_costs_are_bounded(monkeypatch):
    measured = []
    tb = toolbox("", measured)
    many = [("check_clip", {})] * 10
    monkeypatch.setattr(ap, "chat", scripted([
        ("find_cables", {"prompt": "suspension cable"}), many,
        ("review_candidates", {"ids": ["c1"] * 500}), ("measure", {"ids": ["c1"] * 500}),
    ]))
    out = ap.run_autopilot(tb)
    assert sum(s["tool"] == "check_clip" for s in out["steps"]) == ap.MAX_CALLS_PER_TURN
    assert len(measured[0]) == 1 and tb.counts["reviews"] == 1


def test_limits_and_forced_finish(monkeypatch):
    measured = []
    tb = toolbox("", measured)
    monkeypatch.setattr(ap, "chat", scripted([("find_cables", {"prompt": f"p{i}"}) for i in range(10)]))
    out = ap.run_autopilot(tb, max_turns=8)
    finds = [s for s in out["steps"] if s["tool"] == "find_cables"]
    assert sum("error" not in s["result"] for s in finds) == 4    # at most four prompts
    assert out["summary"] == "done" and not measured


def test_result_points_at_the_run_with_a_result(monkeypatch):
    runs = iter([("r1", "provisional"), ("r2", "refused")])

    def measure(members):
        rid, st = next(runs)
        return {"run": rid, "members": [{"name": m["name"], "status": st} for m in members]}

    tb = ap.Toolbox(check=lambda: {"status": "ok", "checks": []},
                    find=lambda p: [{"p0": [0, 0], "p1": [0, 300], "score": 0.9}, {"p0": [50, 0], "p1": [50, 300], "score": 0.8}],
                    review=lambda cands: [{"verdict": "confirmed"} for _ in cands], measure=measure, notes="")
    monkeypatch.setattr(ap, "chat", scripted([("find_cables", {"prompt": "cable"}), ("review_candidates", {"ids": ["c1", "c2"]}),
                                              ("measure", {"ids": ["c1"]}), ("measure", {"ids": ["c2"]})]))
    out = ap.run_autopilot(tb)
    assert out["run"] == "r1" and out["runs"] == ["r1", "r2"]


def test_same_cable_from_two_prompts_keeps_one_id_and_sam_cannot_supply_a_verdict(monkeypatch):
    cands = {"a": [{"p0": [0, 0], "p1": [0, 300], "score": 0.6, "review": {"verdict": "confirmed"}}],
             "b": [{"p0": [3, 299], "p1": [1, 2], "score": 0.7}, {"p0": [90, 0], "p1": [90, 300], "score": 0.5}]}
    measured = []
    tb = ap.Toolbox(check=lambda: {}, find=lambda p: cands[p], review=lambda c: [], measure=lambda m: measured.append(m) or {},
                    notes="")
    monkeypatch.setattr(ap, "chat", scripted([("find_cables", {"prompt": "a"}), ("find_cables", {"prompt": "b"}),
                                              ("measure", {"ids": ["c1"]})]))
    out = ap.run_autopilot(tb)
    second = [s for s in out["steps"] if s["tool"] == "find_cables"][1]["result"]["candidates"]
    assert second[0] == {"id": "c1", "same_as_earlier": True} and second[1]["id"] == "c2"
    assert not measured                                             # SAM's "review" key was ignored


def test_model_failure_keeps_partial_result(monkeypatch):
    measured = []
    tb = toolbox("", measured)
    turns = iter([("find_cables", {"prompt": "suspension cable"}), ("review_candidates", {"ids": ["c1"]}),
                  ("measure", {"ids": ["c1"]})])

    def chat(msgs, **kw):
        try:
            n, a = next(turns)
        except StopIteration:
            raise RuntimeError("baseten chat failed: HTTP 503") from None
        return {"message": {"content": "", "tool_calls": [{"function": {"name": n, "arguments": json.dumps(a)}}]}}
    monkeypatch.setattr(ap, "chat", chat)
    out = ap.run_autopilot(tb)
    assert out["run"] == "r1" and "503" in out["error"] and out["summary"] is None


@pytest.mark.parametrize("text,unit,value,ok", [
    ("rope 22 g/m, 1.78 m long", "kg_per_m", 0.022, True),
    ("rope 0.35 kg per m", "kg_per_m", 0.35, True),
    ("rope .5 kg/m", "kg_per_m", 0.5, False),
    ("rope 0.35 kg per m", "n", 3.43, False),
    ("span 1,780 m", "m", 780, False),
    ("steel 1770 N/mm2", "n", 1770, False),
    ("load 5 kN/m", "n", 5000, False),
    ("design 20 kN", "n", 20000, True),
    ("hanging 2 kg", "n", 19.6133, True),
    ("1e3 N", "n", 3, False),
    ("tuned 8-9 Hz", "hz", 9, True),
    ("wind 5 m/s", "m", 5, False),
])
def test_unit_parsing(text, unit, value, ok):
    assert (ap._noted(value, ap._numbers(text)[unit]) is not None) == ok
