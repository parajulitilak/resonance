"""resonance synth | analyze"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import synth
from .analyze import analyze, compare_with_truth, members_from_truth, save
from .members import parse_member


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="resonance")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="render the synthetic bench rig with known tensions")
    s.add_argument("out")
    s.add_argument("--fps", type=float, default=240.0)
    s.add_argument("--duration", type=float, default=5.0)
    s.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("analyze", help="measure members in a video")
    a.add_argument("video")
    a.add_argument("--member", action="append", default=[],
                   help="name:x0,y0,x1,y1[:L=metres][:m=kg_per_m][:w=half_width_px]")
    a.add_argument("--truth", help="ground-truth json from `synth` (defines members and scores the result)")
    a.add_argument("--anchor", help="static box x0,y0,x1,y1 for shake removal")
    a.add_argument("--capture-fps", type=float)
    a.add_argument("--out", default="out/run")
    a.add_argument("--hero", action="store_true", help="also render the hero video (out/hero.mp4)")
    a.add_argument("--speed", type=float, default=0.25, help="hero playback speed (0.25 = 4x slow motion)")

    i = sub.add_parser("inspect", help="agent review + independent verification of an analysed run")
    i.add_argument("run_dir")
    i.add_argument("--context", default="", help="text, or @file with field notes / design values")
    i.add_argument("--mode", choices=["bench", "field"], default="bench")
    i.add_argument("--inspector", choices=["baseten", "meta"], default="baseten")
    i.add_argument("--verifier", choices=["baseten", "meta"], default="baseten")

    ap_ = sub.add_parser("approve", help="engineer sign-off on an inspection report")
    ap_.add_argument("run_dir")
    ap_.add_argument("--by", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        from .agent import inspect
        ctx = Path(args.context[1:]).read_text() if args.context.startswith("@") else args.context
        rep = json.loads((Path(args.run_dir) / "report.json").read_text())
        md = inspect(rep, ctx, args.run_dir, mode=args.mode, inspector=args.inspector, verifier=args.verifier)
        print(md.read_text())
        return
    if args.cmd == "approve":
        from .agent import approve
        print(approve(args.run_dir, args.by).read_text())
        return
    if args.cmd == "synth":
        t = synth.render(args.out, fps=args.fps, duration=args.duration, seed=args.seed)
        print(json.dumps({"video": args.out, "frames": t["frames"],
                          "tensions_n": [round(r["tension_n"], 2) for r in t["ropes"]]}))
        return

    truth = json.loads(Path(args.truth).read_text()) if args.truth else None
    members = [parse_member(m) for m in args.member] or (members_from_truth(truth) if truth else [])
    if not members:
        ap.error("give --member or --truth")
    anchor = tuple(int(v) for v in args.anchor.split(",")) if args.anchor else (
        tuple(truth["anchor_box"]) if truth else None)
    rep = analyze(args.video, members, anchor=anchor, capture_fps=args.capture_fps)
    if truth:
        rep["truth_comparison"] = compare_with_truth(rep, truth)
    p = save(rep, args.out)
    if args.hero:
        from .displacement import track
        from .render import render_hero
        tr = track(args.video, members, anchor=anchor, capture_fps=args.capture_fps)
        print("hero:", render_hero(args.video, tr, rep, Path(args.out) / "hero.mp4", speed=args.speed))
    for m in rep["members"]:
        lad, ten = m["ladder"], m.get("tension") or {}
        val = ten.get("value_n")
        print(f"{m['name']:>8}  f0={lad['f0_hz']:.3f} Hz  modes={[x['n'] for x in lad['modes']]}  "
              f"ladder={lad['status'].upper()}  "
              + (f"T={val:.1f}±{ten['sigma_n']:.1f} N" if val else ten.get("note", "")))
    if truth:
        for r in rep["truth_comparison"]:
            e = "n/a" if r["err"] is None else f"{r['err']:+.2%}"
            print(f"   truth {r['name']}: {r['true_n']:.1f} N  error {e}")
    print(f"report: {p}")


if __name__ == "__main__":
    main()
