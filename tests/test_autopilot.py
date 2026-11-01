import json

from resonance import autopilot as ap


def scripted(calls):
    """A fake model that makes the given tool calls, one turn each, then finishes."""
    seq = list(calls)

    def fake_chat(msgs, tools=None, provider="baseten", tool_choice=None, **kw):
        name, args = seq.pop(0) if seq and not tool_choice else ("finish", {"summary": "done"})
        call = {"id": f"c{len(msgs)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        return {"message": {"content": "", "tool_calls": [call]}, "usage": {}, "model": "fake"}
    return fake_chat


def toolbox(notes, measured):
    found = [{"p0": [10, 10], "p1": [10, 400], "half_width": 8, "score": 0.6},
             {"p0": [50, 300], "p1": [300, 310], "half_width": 8, "score": 0.5}]
    return ap.Toolbox(
        check=lambda: {"status": "ok", "summary": "fine", "checks": []},
        find=lambda prompt: found if prompt == "suspension cable" else [],
        review=lambda cands: [{"verdict": "confirmed" if c["p0"] == [10, 10] else "rejected", "what": "x"} for c in cands],
        measure=lambda members: measured.append(members) or {"run": "r1", "members": [{"name": m["name"]} for m in members]},
        notes=notes)


def test_agent_cannot_measure_rejected_candidates_or_use_invented_priors(monkeypatch):
    measured = []
    tb = toolbox("Footbridge hangers, design tension about 20 kN; first mode tuned near 5 Hz.", measured)
    monkeypatch.setattr(ap, "chat", scripted([
        ("check_clip", {}),
        ("find_cables", {"prompt": "cable"}),                       # nothing: the agent should try another prompt
        ("find_cables", {"prompt": "suspension cable"}),
        ("review_candidates", {"ids": ["c1", "c2"]}),
        ("measure", {"ids": ["c2"]}),                               # rejected by the review: blocked
        ("measure", {"ids": ["c1", "c2"], "f0_range_hz": [4, 6], "length_m": 12.0}),
    ]))
    out = ap.run_autopilot(tb)
    blocked = next(s for s in out["steps"] if s["tool"] == "measure")
    assert "error" in blocked["result"] and blocked["result"]["not_confirmed"] == ["c2"]
    assert len(measured) == 1 and [m["name"] for m in measured[0]] == ["c1"]
    assert measured[0][0]["f0_range"] == [4, 6]                     # 5 Hz is in the notes
    assert "length_m" not in measured[0][0]                         # 12 m is not: dropped
    assert out["run"] == "r1" and out["summary"] == "done"


def test_limits_and_forced_finish(monkeypatch):
    measured = []
    tb = toolbox("", measured)
    monkeypatch.setattr(ap, "chat", scripted([("find_cables", {"prompt": f"p{i}"}) for i in range(10)]))
    out = ap.run_autopilot(tb, max_turns=8)
    finds = [s for s in out["steps"] if s["tool"] == "find_cables"]
    assert sum("error" not in s["result"] for s in finds) == 4    # at most four prompts
    assert out["summary"] == "done" and not measured


def test_grounding_rules_are_unit_aware():
    nums = ap._numbers("rope 22 g/m, 1.78 m long, hanging 2 kg; tuned 8-9 Hz; design 20 kN")
    assert ap._grounded(0.022, nums["kg_per_m"]) and ap._grounded(1.78, nums["m"])
    assert ap._grounded([7.5, 9.5], nums["hz"]) and not ap._grounded([20, 30], nums["hz"])  # 22 is g/m, not Hz
    assert ap._grounded([15, 25], nums["n"])          # 2 kg hanging -> 19.6 N
    assert ap._grounded([15000, 25000], nums["n"])    # 20 kN
    assert not ap._grounded(1.78, nums["kg_per_m"]) and not ap._grounded([9, 8], nums["hz"])


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
