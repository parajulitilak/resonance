"""One take in, one evidence-backed report out."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .displacement import Tracking, track
from .members import Member
from .physics import tension
from .spectrum import Peak, count_dips, fit_harmonics, local_peaks, mode_magnitude, peaks, spectrum


def _shake_peaks(tr: Tracking) -> list[float]:
    """Frequencies where the camera itself moved (hand tremor, stabiliser): not cable modes."""
    if tr.shake is None:
        return []
    sp = spectrum(tr.t, tr.shake - tr.shake.mean(0))
    return [p.f for p in peaks(sp, min_snr=10.0)]


def analyze_tracking(tr: Tracking) -> dict:
    tb = tr.timebase
    shake_f = _shake_peaks(tr)
    report = {
        "timing": {"fps": tb.fps, "nominal_fps": tb.nominal_fps, "jitter": tb.jitter,
                   "variable_frame_rate": tb.variable, "ramped": tb.ramped, "time_scale": tb.time_scale,
                   "source": tb.source, "frames": tr.n_frames, "duration_s": float(tr.t[-1] - tr.t[0])},
        "shake_px": None if tr.shake is None else {
            "rms": float(np.sqrt(np.mean(np.sum((tr.shake - tr.shake.mean(0)) ** 2, 1)))),
            "max": float(np.abs(tr.shake - tr.shake.mean(0)).max()),
            "peaks_hz": [round(f, 3) for f in shake_f]},
        "members": [],
    }
    duration = float(tr.t[-1] - tr.t[0])
    all_peaks: dict[str, list[Peak]] = {}  # cable members only
    for m in tr.members:
        w = tr.weights[m.name] if tr.weights else None
        sp = spectrum(tr.t, tr.disp[m.name], weights=w, pad=8 if m.kind == "patch" else 4)
        pk_all = peaks(sp, f_min=0.1 if m.kind == "patch" else 0.3, min_snr=5.0 if m.kind == "patch" else 8.0)
        if m.kind == "patch" and m.length_m and m.mass_per_m:
            # a tracked point on a string of known length and mass: full ladder check and tension physics,
            # but no mode-shape check (one point can't show nodes)
            pk = [p for p in pk_all if not any(abs(p.f - f) <= 0.01 * f for f in shake_f)]
            prior = m.expected_f0()
            fit = fit_harmonics(pk, sp.fs, f0_range=prior, duration=duration)
            entry = {"name": m.name, "kind": "patch", "box": list(m.box), "direction": list(m.direction),
                     "quality": tr.quality[m.name], "prior_f0_hz": list(prior) if prior else None,
                     "peaks": [asdict(p) for p in pk],
                     "ladder": {"ok": fit.ok, "status": fit.status, "reasons": fit.reasons, "f0_hz": fit.f0,
                                "f0_sigma_hz": fit.f0_sigma, "b": fit.b, "rel_residual": fit.rel_residual,
                                "explained": fit.explained,
                                "modes": [{"n": n, "f_hz": f, "snr": s_} for n, f, s_ in fit.modes]},
                     "mode_nodes": {}, "notes": ["single tracked point: mode-shape check not available"]}
            if fit.status in ("verified", "provisional"):
                est = tension(fit.f0, fit.f0_sigma, fit.b, m.length_m, m.mass_per_m)
                entry["tension"] = {"value_n": est.tension_n, "sigma_n": est.sigma_n, "xi": est.xi,
                                    "clamped_n": est.clamped_n, "reliable": est.reliable and fit.ok,
                                    "provisional": fit.status == "provisional", "note": est.note,
                                    "inputs": {"length_m": m.length_m, "mass_per_m": m.mass_per_m}}
            else:
                entry["tension"] = {"value_n": None, "note": "refused: " + "; ".join(fit.reasons)}
            report["members"].append(entry)
            continue
        if m.kind == "patch":
            pk = [p for p in local_peaks(sp, f_min=0.1) if not any(abs(p.f - f) <= 0.01 * f for f in shake_f)]
            dom = max(pk, key=lambda p: p.power) if pk else None
            report["members"].append({
                "name": m.name, "kind": "patch", "box": list(m.box), "direction": list(m.direction),
                "quality": tr.quality[m.name], "peaks": [asdict(p) for p in pk],
                "ladder": {"ok": False, "status": "frequency" if dom else "refused",
                           "reasons": [] if dom else ["no clear vibration peak in the patch motion"],
                           "f0_hz": dom.f if dom else float("nan"),
                           "f0_sigma_hz": (0.05 / duration) if dom else float("nan"), "b": 0.0,
                           "rel_residual": float("nan"), "explained": 0.0, "modes": []},
                "mode_nodes": {}, "tension": {"value_n": None, "note": "patch: frequencies only, no tension physics"}})
            continue
        # drop peaks the camera made (they appear in the anchor's own motion)
        pk = [p for p in pk_all if not any(abs(p.f - f) <= max(0.01 * f, 2 * (sp.f[1] - sp.f[0])) for f in shake_f)]
        all_peaks[m.name] = pk
        prior = m.expected_f0()
        fit = fit_harmonics(pk, sp.fs, f0_range=prior, duration=duration)
        reasons = list(fit.reasons)
        status = fit.status
        if tb.ramped:
            status, reasons = "refused", reasons + [
                "frame timing changes during the clip (an exported slow-motion file with real-time ramps?): "
                "use the original recording"]
        # Mode shapes from amplitudes alone: rung 1 must have no interior node, rung 2 exactly one.
        nodes = {}
        if status != "refused":
            for n, f, _ in fit.modes:
                if n <= 2:
                    nodes[str(n)] = count_dips(mode_magnitude(sp, f))
            bad = [n for n, k in nodes.items() if k != int(n) - 1]
            if bad:
                status = "refused"
                reasons.append(f"mode-shape check failed: rung {bad[0]} shows {nodes[bad[0]]} node(s), expected "
                               f"{int(bad[0]) - 1} (the 'fundamental' may be a higher mode, or the member is "
                               "only partly in view)")
        entry = {
            "name": m.name, "p0": list(m.p0), "p1": list(m.p1), "length_px": m.length_px,
            "quality": tr.quality[m.name], "prior_f0_hz": list(prior) if prior else None,
            "peaks": [asdict(p) for p in pk], "camera_peaks_removed": len(pk_all) - len(pk),
            "ladder": {"ok": status == "verified", "status": status, "reasons": reasons, "f0_hz": fit.f0,
                       "f0_sigma_hz": fit.f0_sigma, "b": fit.b, "rel_residual": fit.rel_residual,
                       "explained": fit.explained,
                       "modes": [{"n": n, "f_hz": f, "snr": s} for n, f, s in fit.modes]},
            "mode_nodes": nodes,
        }
        if status in ("verified", "provisional") and m.length_m and m.mass_per_m:
            est = tension(fit.f0, fit.f0_sigma, fit.b, m.length_m, m.mass_per_m)
            entry["tension"] = {"value_n": est.tension_n, "sigma_n": est.sigma_n, "xi": est.xi,
                                "clamped_n": est.clamped_n, "reliable": est.reliable and status == "verified",
                                "provisional": status == "provisional",
                                "note": est.note if status == "verified" else
                                "provisional: " + "; ".join(reasons),
                                "inputs": {"length_m": m.length_m, "mass_per_m": m.mass_per_m}}
        elif status == "refused":
            entry["tension"] = {"value_n": None, "note": "refused: " + "; ".join(reasons)}
        else:
            entry["tension"] = {"value_n": None, "note": f"{status}: length and mass per metre not given"}
        report["members"].append(entry)
    # Peaks shared by most members at once are structure-wide (deck, frame) or camera motion: flag them.
    names = list(all_peaks)
    if len(names) >= 3:
        common = []
        for p in all_peaks[names[0]]:
            hits = sum(any(abs(q.f - p.f) <= 0.005 * p.f for q in all_peaks[nm]) for nm in names)
            if hits >= max(3, int(0.6 * len(names))):
                common.append(round(p.f, 3))
        report["common_peaks_hz"] = common
    return report


def analyze(path: str, members: list[Member], anchor=None, capture_fps: float | None = None,
            max_frames: int | None = None) -> dict:
    tr = track(path, members, anchor=anchor, capture_fps=capture_fps, max_frames=max_frames)
    rep = analyze_tracking(tr)
    rep["video"] = str(path)
    return rep


def members_from_truth(truth: dict, tension_range: tuple[float, float] | None = (5.0, 300.0)) -> list[Member]:
    """Members for a synthetic rig. The tension range is a deliberately coarse prior (the alias guard)."""
    return [Member(r["name"], tuple(r["p0"]), tuple(r["p1"]), half_width=9.0, length_m=r["length_m"],
                   mass_per_m=r["mass_per_m"], tension_range=tension_range) for r in truth["ropes"]]


def compare_with_truth(report: dict, truth: dict) -> list[dict]:
    rows = []
    by = {m["name"]: m for m in report["members"]}
    for r in truth["ropes"]:
        m = by.get(r["name"], {})
        t = (m.get("tension") or {}).get("value_n")
        rows.append({"name": r["name"], "true_n": r["tension_n"], "est_n": t,
                     "err": None if t is None else (t - r["tension_n"]) / r["tension_n"],
                     "true_f0": r["f0_hz"], "est_f0": m.get("ladder", {}).get("f0_hz"),
                     "status": m.get("ladder", {}).get("status"),
                     "ladder_ok": m.get("ladder", {}).get("ok")})
    return rows


def save(report: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "report.json"
    p.write_text(json.dumps(report, indent=2, default=float))
    return p
