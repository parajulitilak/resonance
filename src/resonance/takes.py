"""Second-take confirmation: the alias guard that needs no prior.

An aliased peak lands at |f - k fs|, so it moves when the frame rate changes; a
real mode doesn't. If two takes of the same member at clearly different frame
rates give the same ladder, aliasing is ruled out and a provisional result can
be verified. If they disagree, at least one take is aliased and both are refused.
"""

from __future__ import annotations

import copy

from .physics import tension


def confirm(rep_a: dict, rep_b: dict, min_rate_ratio: float = 1.2, tol: float = 0.01) -> dict:
    """Merge two analysed takes of the same members into one report (take A's numbers, cross-checked)."""
    fa, fb = rep_a["timing"]["fps"], rep_b["timing"]["fps"]
    out = copy.deepcopy(rep_a)
    out["second_take"] = {"video": rep_b.get("video"), "fps": fb}
    rates_differ = max(fa, fb) / min(fa, fb) >= min_rate_ratio
    by_b = {m["name"]: m for m in rep_b["members"]}
    for m in out["members"]:
        other = by_b.get(m["name"])
        lad = m["ladder"]
        check = {"other_status": None, "other_f0_hz": None, "agree": None}
        m["cross_take"] = check
        if other is None:
            continue
        lb = other["ladder"]
        check.update(other_status=lb["status"], other_f0_hz=lb["f0_hz"])
        # take A must carry a full ladder (>= 3 rungs); take B only has to reproduce the same fundamental
        if not (lad["status"] in ("verified", "provisional") and len(lad["modes"]) >= 3
                and lb["status"] in ("verified", "provisional") and len(lb["modes"]) >= 2):
            continue
        sig = (lad["f0_sigma_hz"] ** 2 + lb["f0_sigma_hz"] ** 2) ** 0.5
        agree = abs(lad["f0_hz"] - lb["f0_hz"]) <= max(tol * lad["f0_hz"], 3 * sig)
        check["agree"] = bool(agree)
        if not agree:
            lad["status"], lad["ok"] = "refused", False
            lad["reasons"] = lad["reasons"] + [f"second take at {fb:.0f} fps gives f0 {lb['f0_hz']:.3f} Hz, "
                                               f"not {lad['f0_hz']:.3f} Hz: at least one take is aliased"]
            m["tension"] = {"value_n": None, "note": "refused: " + "; ".join(lad["reasons"])}
            continue
        if not rates_differ:
            check["note"] = "same frame rate: repeatability only, not an alias check"
            continue
        only_guard = lad["status"] == "provisional" and all(r.startswith("no frequency prior") for r in lad["reasons"])
        if only_guard:
            lad["status"], lad["ok"] = "verified", True
            lad["reasons"] = [f"alias guard: confirmed by a second take at {fb:.0f} fps"]
            ten = m.get("tension") or {}
            inp = ten.get("inputs") or {}
            if inp.get("length_m") and inp.get("mass_per_m"):
                est = tension(lad["f0_hz"], lad["f0_sigma_hz"], lad["b"], inp["length_m"], inp["mass_per_m"])
                m["tension"] = {"value_n": est.tension_n, "sigma_n": est.sigma_n, "xi": est.xi,
                                "clamped_n": est.clamped_n, "reliable": est.reliable, "provisional": False,
                                "note": est.note, "inputs": inp}
    return out
