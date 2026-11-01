import numpy as np

from resonance import review as rv
from resonance.members import Member

FRAME = np.zeros((120, 160, 3), np.uint8)
M = Member("h1", (80, 10), (80, 110))


def _reply(text):
    return lambda *a, **k: {"message": {"content": text}, "model": "test"}


def test_confirmed_and_rejected_are_read(monkeypatch):
    monkeypatch.setattr(rv, "chat", _reply('Sure. {"is_tensioned_member": true, "what_it_is": "hanger", '
                                           '"both_supports_visible": true, "reason": "taut"}'))
    assert rv.review_member(FRAME, M)["verdict"] == "confirmed"
    monkeypatch.setattr(rv, "chat", _reply('{"is_tensioned_member": false, "what_it_is": "floor lead", "reason": "on the ground"}'))
    r = rv.review_member(FRAME, M)
    assert r["verdict"] == "rejected" and r["what"] == "floor lead"


def test_failures_are_unreviewed_never_confirmed(monkeypatch):
    for text in ["", "yes it is a cable", '{"is_tensioned_member": "yes"}', '{"is_tensioned_member": true']:
        monkeypatch.setattr(rv, "chat", _reply(text))
        assert rv.review_member(FRAME, M)["verdict"] == "unreviewed", text

    def boom(*a, **k):
        raise RuntimeError("baseten chat failed: HTTP 503")
    monkeypatch.setattr(rv, "chat", boom)
    r = rv.review_member(FRAME, M)
    assert r["verdict"] == "unreviewed" and "503" in r["reason"]


def test_review_keeps_member_order(monkeypatch):
    monkeypatch.setattr(rv, "chat", _reply('{"is_tensioned_member": true}'))
    ms = [Member(f"m{i}", (10 * i, 0), (10 * i, 100)) for i in range(1, 6)]
    assert [r["name"] for r in rv.review(FRAME, ms)] == ["m1", "m2", "m3", "m4", "m5"]


def test_server_json_has_no_nan():
    import json
    import math

    from resonance.server import _finite
    rep = {"members": [{"ladder": {"f0_hz": math.nan, "f0_sigma_hz": math.inf, "modes": []}, "peaks": [1.5]}]}
    out = json.dumps(_finite(rep), allow_nan=False)
    assert '"f0_hz": null' in out and '"f0_sigma_hz": null' in out and "1.5" in out
