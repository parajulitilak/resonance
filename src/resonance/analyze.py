"""One take in, one evidence-backed report out."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .displacement import Tracking, track
from .members import Member
from .physics import tension
from .spectrum import count_nodes, fit_harmonics, mode_shape, peaks, spectrum


def analyze_tracking(tr: Tracking) -> dict:
    tb = tr.timebase
    report = {
        "timing": {"fps": tb.fps, "nominal_fps": tb.nominal_fps, "jitter": tb.jitter,
                   "variable_frame_rate": tb.variable, "time_scale": tb.time_scale, "source": tb.source,
                   "frames": tr.n_frames, "duration_s": float(tr.t[-1] - tr.t[0])},
        "shake_px": None if tr.shake is None else {
            "rms": float(np.sqrt(np.mean(np.sum((tr.shake - tr.shake.mean(0)) ** 2, 1)))),
            "max": float(np.abs(tr.shake - tr.shake.mean(0)).max())},
        "members": [],
    }
    for m in tr.members:
        sp = spectrum(tr.t, tr.disp[m.name])
        pk = peaks(sp)
        fit = fit_harmonics(pk, sp.fs)
        entry = {
            "name": m.name, "p0": list(m.p0), "p1": list(m.p1), "length_px": m.length_px,
            "quality": tr.quality[m.name],
            "peaks": [asdict(p) for p in pk],
            "ladder": {"ok": fit.ok, "status": fit.status, "reasons": fit.reasons, "f0_hz": fit.f0, "f0_sigma_hz": fit.f0_sigma,
                       "b": fit.b, "rel_residual": fit.rel_residual, "explained": fit.explained,
                       "modes": [{"n": n, "f_hz": f, "snr": s} for n, f, s in fit.modes]},
        }
        if fit.ok:
            entry["mode_nodes"] = {str(n): count_nodes(mode_shape(sp, f)) for n, f, _ in fit.modes[:4]}
        if (fit.ok or fit.provisional) and m.length_m and m.mass_per_m:
            est = tension(fit.f0, fit.f0_sigma, fit.b, m.length_m, m.mass_per_m)
            entry["tension"] = {"value_n": est.tension_n, "sigma_n": est.sigma_n, "xi": est.xi,
                                "reliable": est.reliable and fit.ok, "provisional": fit.provisional,
                                "note": est.note if fit.ok else "provisional: only 2 harmonics; re-shoot to confirm",
                                "inputs": {"length_m": m.length_m, "mass_per_m": m.mass_per_m}}
        elif not fit.ok:
            entry["tension"] = {"value_n": None, "note": "refused: " + "; ".join(fit.reasons)}
        report["members"].append(entry)
    return report


def analyze(path: str, members: list[Member], anchor=None, capture_fps: float | None = None,
            max_frames: int | None = None) -> dict:
    tr = track(path, members, anchor=anchor, capture_fps=capture_fps, max_frames=max_frames)
    rep = analyze_tracking(tr)
    rep["video"] = str(path)
    return rep


def members_from_truth(truth: dict) -> list[Member]:
    return [Member(r["name"], tuple(r["p0"]), tuple(r["p1"]), half_width=9.0,
                   length_m=r["length_m"], mass_per_m=r["mass_per_m"]) for r in truth["ropes"]]


def compare_with_truth(report: dict, truth: dict) -> list[dict]:
    rows = []
    by = {m["name"]: m for m in report["members"]}
    for r in truth["ropes"]:
        m = by.get(r["name"], {})
        t = (m.get("tension") or {}).get("value_n")
        rows.append({"name": r["name"], "true_n": r["tension_n"], "est_n": t,
                     "err": None if t is None else (t - r["tension_n"]) / r["tension_n"],
                     "true_f0": r["f0_hz"], "est_f0": m.get("ladder", {}).get("f0_hz"),
                     "ladder_ok": m.get("ladder", {}).get("ok")})
    return rows


def save(report: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "report.json"
    p.write_text(json.dumps(report, indent=2, default=float))
    return p
