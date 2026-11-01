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


def _thumb(sp, pk, bins: int = 240) -> dict:
    """A small copy of the spectrum for the report page: highest dB above the noise floor in each bin."""
    hi = sp.fs / 2 if not pk else min(sp.fs / 2, max(2.0, 1.25 * max(p.f for p in pk)))
    sel = (sp.f > 0) & (sp.f <= hi)
    db = 10 * np.log10(np.maximum(sp.power[sel] / np.maximum(sp.floor[sel], 1e-30), 1.0))
    idx = np.minimum((sp.f[sel] / hi * bins).astype(int), bins - 1)
    out = np.zeros(bins)
    np.maximum.at(out, idx, db)
    return {"f_max_hz": float(hi), "snr_db": [round(float(x), 1) for x in out]}


def analyze_tracking(tr: Tracking, picture_motion_px: float | None = None) -> dict:
    """`picture_motion_px`: whole-frame motion RMS (clipcheck.global_shake), used for the camera gate when no
    static anchor was marked."""
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
    # Camera gate: translation is removed, but large camera motion also brings rotation, zoom and parallax.
    camera_reason = None
    if report["shake_px"] and report["shake_px"]["rms"] > 5.0:
        camera_reason = (f"camera moved {report['shake_px']['rms']:.0f} px RMS (max {report['shake_px']['max']:.0f} px): "
                         "too much to correct; film from a tripod or rest the phone on something solid")
    elif tr.shake is None and picture_motion_px is not None:
        report["picture_motion_px"] = picture_motion_px
        if not np.isfinite(picture_motion_px):  # could not be judged: never treat as still
            camera_reason = ("camera motion could not be checked (too little static texture away from the cables) and "
                             "no static anchor was marked: mark a static, textured anchor or film again")
        elif picture_motion_px > 5.0:  # no anchor to correct with: the camera's own motion would look like vibration
            camera_reason = (f"the whole picture moved {picture_motion_px:.0f} px RMS and no static anchor was marked: "
                             "camera motion cannot be told apart from cable motion; brace the phone and film again")
    report["camera_gate"] = camera_reason
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
            gate = [r for r in (camera_reason, "frame timing changes during the clip: use the original recording"
                                if tb.ramped else None) if r]
            if gate:  # same gates as cable members
                fit.ok, fit.provisional, fit.reasons = False, False, list(fit.reasons) + gate
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
            entry["spectrum"] = _thumb(sp, pk)
            report["members"].append(entry)
            continue
        if m.kind == "patch":
            pk = [p for p in local_peaks(sp, f_min=0.1) if not any(abs(p.f - f) <= 0.01 * f for f in shake_f)]
            dom = max(pk, key=lambda p: p.power) if pk else None
            q = tr.quality[m.name]
            gate = []
            if camera_reason:
                gate.append(camera_reason)
            # frame-to-frame match, 10th percentile. Calibrated on 4 clips (tree 0.60, cello 0.36 valid;
            # runners crossing a deck 0.24-0.27): provisional threshold
            if q.get("match_p10", 1.0) < 0.30:
                gate.append(f"the tracked patch changes between frames (match {q['match_p10']:.2f} at the 10th "
                            "percentile, below 0.30): people or objects pass in front of it")
            if gate:
                dom = None
            report["members"].append({
                "name": m.name, "kind": "patch", "box": list(m.box), "direction": list(m.direction),
                "quality": tr.quality[m.name], "peaks": [asdict(p) for p in pk],
                "ladder": {"ok": False, "status": "frequency" if dom else "refused",
                           "reasons": [] if dom else (gate or ["no clear vibration peak in the patch motion"]),
                           "f0_hz": dom.f if dom else float("nan"),
                           "f0_sigma_hz": (0.05 / duration) if dom else float("nan"), "b": 0.0,
                           "rel_residual": float("nan"), "explained": 0.0, "modes": []},
                "mode_nodes": {}, "tension": {"value_n": None, "note": "patch: frequencies only, no tension physics"},
                "spectrum": _thumb(sp, pk)})
            continue
        # drop peaks the camera made (they appear in the anchor's own motion)
        pk = [p for p in pk_all if not any(abs(p.f - f) <= max(0.01 * f, 2 * (sp.f[1] - sp.f[0])) for f in shake_f)]
        all_peaks[m.name] = pk
        prior = m.expected_f0()
        fit = fit_harmonics(pk, sp.fs, f0_range=prior, duration=duration)
        reasons = list(fit.reasons)
        status = fit.status
        single_mode = None
        if status == "refused" and pk and prior and not camera_reason and not tb.ramped:
            # One dominant mode and no ladder (e.g. wind lock-in): identify the mode number from its shape
            # (n - 1 interior nodes), then f0 = f / n. Allowed only if mode n can't be aliased given the prior.
            ranked = sorted(pk, key=lambda p: -p.power)
            dom = ranked[0]
            second = ranked[1].power if len(ranked) > 1 else 0.0
            n_mode = count_dips(mode_magnitude(sp, dom.f)) + 1
            f0_est = dom.f / n_mode
            if (dom.snr >= 50 and dom.power >= 2 * second and n_mode <= 6 and prior[0] <= f0_est <= prior[1]
                    and n_mode * prior[1] <= sp.fs / 2.5 * 1.001):
                single_mode = (n_mode, dom)
                status = "provisional"
                fit.f0, fit.b = f0_est, 0.0
                fit.modes = [(n_mode, dom.f, dom.snr)]
                fit.sigma_override = max(0.05 / duration, 1e-4 * dom.f) / n_mode
                reasons = [f"single mode identified from its shape: {n_mode - 1} node(s) → mode {n_mode} at "
                           f"{dom.f:.2f} Hz, so f0 = {f0_est:.3f} Hz (no harmonic ladder to confirm it)"]
        if camera_reason:
            status, reasons = "refused", reasons + [camera_reason]
        if tb.ramped:
            status, reasons = "refused", reasons + [
                "frame timing changes during the clip (an exported slow-motion file with real-time ramps?): "
                "use the original recording"]
        # Mode shapes from amplitudes alone: rung 1 must have no interior node, rung 2 exactly one.
        nodes = {}
        if single_mode:
            nodes = {str(single_mode[0]): single_mode[0] - 1}
        elif status != "refused":
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
        entry["spectrum"] = _thumb(sp, pk)
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


def picture_motion(path: str, members: list[Member] | None = None) -> float:
    """Camera motion RMS (px) away from the members themselves (they are meant to move); NaN if it cannot be
    judged, which the camera gate treats as a refusal."""
    from .clipcheck import global_shake
    sh = global_shake(path, exclude=[_region(m) for m in members or []])
    if sh is None or len(sh) < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((sh - sh.mean(0)) ** 2, 1))))


def _region(m: Member, pad: float = 20.0):
    """What to leave out of the camera estimate: a patch's box, or a band along a cable's line (a bounding box
    of a diagonal cable would cover most of the frame)."""
    if m.kind == "patch" and m.box:
        x0, y0, x1, y1 = m.box
        return x0 - pad, y0 - pad, x1 + pad, y1 + pad
    return tuple(m.p0), tuple(m.p1), m.half_width + pad


def analyze(path: str, members: list[Member], anchor=None, capture_fps: float | None = None,
            max_frames: int | None = None) -> dict:
    tr = track(path, members, anchor=anchor, capture_fps=capture_fps, max_frames=max_frames)
    rep = analyze_tracking(tr, picture_motion_px=None if anchor else picture_motion(path, members))
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


def _commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "describe", "--always", "--dirty",
                               "--tags"], capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save(report: dict, out_dir: str | Path, log: str | Path = "out/runs.jsonl") -> Path:
    """Write report.json and append one line per member to the run log (for later evaluation)."""
    import hashlib
    import time
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.setdefault("code_version", _commit())
    p = out / "report.json"
    p.write_text(json.dumps(report, indent=2, default=float))
    video = report.get("video")
    sha = None
    if video and Path(video).exists() and Path(video).stat().st_size < 2_000_000_000:
        sha = hashlib.sha256(Path(video).read_bytes()).hexdigest()
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        for m in report.get("members", []):
            lad, ten = m["ladder"], m.get("tension") or {}
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": str(out), "video": video,
                                "video_sha256": sha, "code": report["code_version"], "fps": report["timing"]["fps"],
                                "member": m["name"], "kind": m.get("kind", "cable"), "status": lad["status"],
                                "f0_hz": lad["f0_hz"], "f0_sigma_hz": lad["f0_sigma_hz"], "rungs": len(lad["modes"]),
                                "tension_n": ten.get("value_n"), "reasons": lad["reasons"]}, default=float) + "\n")
    return p
