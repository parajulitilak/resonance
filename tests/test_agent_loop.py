import json

from resonance import agent

REPORT = {"timing": {"fps": 240.0}, "members": [
    {"name": "rope1", "ladder": {"status": "verified", "ok": True, "f0_hz": 8.4, "f0_sigma_hz": 0.01, "reasons": [],
                                 "modes": [{"n": 1, "f_hz": 8.4, "snr": 50}]},
     "tension": {"value_n": 19.6, "sigma_n": 0.4}, "peaks": [], "quality": {}}]}


def test_agent_that_keeps_flagging_is_made_to_submit(monkeypatch):
    seen = {"nudged": False, "forced": 0}

    def fake_chat(msgs, tools=None, provider="baseten", tool_choice=None, **kw):
        seen["nudged"] |= any("submit_report now" in str(m.get("content")) for m in msgs if m["role"] == "user")
        if tool_choice:
            seen["forced"] += 1
            name, args = "submit_report", {"summary": "s", "members": [
                {"name": "rope1", "status": "verified", "statement": "Tension 19.6 ± 0.4 N.", "evidence": []}]}
        else:  # a model stuck re-flagging the same member
            name, args = "flag", {"name": "rope1", "severity": "info", "finding": "again", "evidence": []}
        call = {"id": f"c{len(msgs)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        return {"message": {"content": "", "tool_calls": [call]}, "usage": {}, "model": "fake"}

    monkeypatch.setattr(agent, "chat", fake_chat)
    acts = agent.run_inspector(REPORT, "", log=lambda *a: None, max_turns=8)
    assert acts["report"] and acts["report"]["members"][0]["name"] == "rope1"
    assert len(acts["flags"]) == 1 and seen["nudged"] and seen["forced"] == 1


def test_verifier_batches_long_reports_and_answers_every_claim(monkeypatch):
    sizes = []

    def fake_chat(msgs, provider="baseten", max_tokens=0, **kw):
        claims = json.loads(msgs[1]["content"].split("CLAIMS:\n", 1)[1])
        sizes.append(len(claims))
        return {"message": {"content": json.dumps([{"id": c["id"], "verdict": "agree", "reason": "ok"} for c in claims])},
                "usage": {}, "model": "fake"}

    monkeypatch.setattr(agent, "chat", fake_chat)
    rep = {"timing": {}, "members": [{"name": f"m{i}", "peaks": [], "ladder": {}, "quality": {}} for i in range(20)]}
    acts = {"report": {"members": [{"name": f"m{i}", "status": "refused", "statement": "s"} for i in range(20)]}, "flags": []}
    v = agent.run_verifier(rep, acts)
    assert len(v) == 20 and all(x["verdict"] == "agree" for x in v)
    assert max(sizes) <= 6 and sum(sizes) == 20
