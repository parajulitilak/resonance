from resonance.fingerprint import diff


def _rep(f0s, status="verified", sig=0.01):
    return {"members": [{"name": f"r{i}", "ladder": {"status": status, "f0_hz": f, "f0_sigma_hz": sig}}
                        for i, f in enumerate(f0s)]}


def test_one_member_changed():
    res = diff(_rep([10.0, 12.0, 14.0]), _rep([10.0, 12.0, 14.0 * (0.75 ** 0.5)]))
    assert res["changed"] == ["r2"]
    row = res["members"][2]
    assert abs(row["dT_rel"] + 0.25) < 1e-9


def test_small_change_below_floor_is_unchanged():
    res = diff(_rep([10.0]), _rep([10.0 * 1.005]))   # +1% tension < 2% floor
    assert res["changed"] == []


def test_refused_is_not_comparable():
    res = diff(_rep([10.0]), _rep([10.0], status="refused"))
    assert res["members"][0]["status"] == "not comparable"
