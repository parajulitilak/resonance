"""Baseline fingerprints and re-checks: what changed since last time?

On a real bridge the absolute tension of a short hanger is uncertain, but its frequency is measured to
a fraction of a percent, and for the same member (same length, same mass) tension scales with f0^2.
So the robust product is a fingerprint of every member, re-checked after a flood, a storm or a repair:
"hanger 7: tension down 18% (± 2%) since the baseline".
"""

from __future__ import annotations

import math


def diff(baseline: dict, recheck: dict, floor: float = 0.02, k_sigma: float = 3.0) -> dict:
    """Compare two analysed takes member by member (matched by name)."""
    base = {m["name"]: m for m in baseline["members"]}
    rows = []
    for m in recheck["members"]:
        b = base.get(m["name"])
        row = {"name": m["name"], "status": "not comparable", "reason": ""}
        if b is None:
            row["reason"] = "no baseline for this member"
            rows.append(row)
            continue
        la, lb = b["ladder"], m["ladder"]
        if la["status"] == "refused" or lb["status"] == "refused":
            row["reason"] = "baseline or re-check refused by the physics check"
            rows.append(row)
            continue
        fa, fb, sa, sb = la["f0_hz"], lb["f0_hz"], la["f0_sigma_hz"], lb["f0_sigma_hz"]
        r = fb / fa
        dt = r * r - 1.0                                    # tension change for the same L and m
        sig_r = r * math.sqrt((sa / fa) ** 2 + (sb / fb) ** 2)
        sig_dt = 2 * r * sig_r
        threshold = max(floor, k_sigma * sig_dt)
        changed = abs(dt) > threshold
        row.update({"f0_baseline_hz": fa, "f0_recheck_hz": fb, "df_rel": r - 1.0, "dT_rel": dt,
                    "dT_sigma": sig_dt, "threshold": threshold,
                    "status": ("changed" if changed else "unchanged"),
                    "provisional": la["status"] == "provisional" or lb["status"] == "provisional"})
        rows.append(row)
    changed = [r for r in rows if r["status"] == "changed"]
    return {"baseline_video": baseline.get("video"), "recheck_video": recheck.get("video"),
            "members": rows, "changed": [r["name"] for r in changed]}


def describe(row: dict) -> str:
    if row["status"] == "not comparable":
        return f"{row['name']}: not comparable ({row['reason']})"
    word = "down" if row["dT_rel"] < 0 else "up"
    tag = " (provisional)" if row.get("provisional") else ""
    if row["status"] == "unchanged":
        return f"{row['name']}: unchanged (tension {row['dT_rel']:+.1%} ± {row['dT_sigma']:.1%}){tag}"
    return f"{row['name']}: tension {word} {abs(row['dT_rel']):.1%} ± {row['dT_sigma']:.1%} since baseline{tag}"
