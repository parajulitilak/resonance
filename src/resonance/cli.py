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
    i.add_argument("--baseline", help="run dir of an earlier take of the same members (change detection)")

    df = sub.add_parser("diff", help="re-check against a baseline take: which members changed?")
    df.add_argument("baseline_run")
    df.add_argument("recheck_run")
    df.add_argument("--out")

    sv = sub.add_parser("serve", help="local web app on http://127.0.0.1:8765")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it from a phone on the same Wi-Fi")

    dt = sub.add_parser("detect", help="find members with SAM 3.1 (on-device if configured) and save an overlay")
    dt.add_argument("video")
    dt.add_argument("--concept", default="rope")
    dt.add_argument("--out", default="out/detect.png")

    ck = sub.add_parser("check", help="check a clip before measuring (true fps, timing, shake, exposure)")
    ck.add_argument("video")
    ck.add_argument("--capture-fps", type=float)

    c = sub.add_parser("confirm", help="cross-check two takes of the same members (second-take alias guard)")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--out", required=True)

    ap_ = sub.add_parser("approve", help="engineer sign-off on an inspection report")
    ap_.add_argument("run_dir")
    ap_.add_argument("--by", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        from .agent import inspect
        ctx = Path(args.context[1:]).read_text() if args.context.startswith("@") else args.context
        rep = json.loads((Path(args.run_dir) / "report.json").read_text())
        base = json.loads((Path(args.baseline) / "report.json").read_text()) if args.baseline else None
        md = inspect(rep, ctx, args.run_dir, mode=args.mode, inspector=args.inspector, verifier=args.verifier,
                     baseline=base)
        print(md.read_text())
        return
    if args.cmd == "diff":
        from .fingerprint import describe, diff
        a = json.loads((Path(args.baseline_run) / "report.json").read_text())
        b = json.loads((Path(args.recheck_run) / "report.json").read_text())
        res = diff(a, b)
        for r in res["members"]:
            print(("⚠ " if r["status"] == "changed" else "  ") + describe(r))
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "diff.json").write_text(json.dumps(res, indent=2, default=float))
        return
    if args.cmd == "serve":
        from .server import serve
        serve(args.port, args.host)
        return
    if args.cmd == "detect":
        import time

        from PIL import Image, ImageDraw

        from . import sam_local
        from .llm import load_env
        from .video import read_frames
        load_env()
        frame = next(iter(read_frames(args.video)))
        t0 = time.time()
        if sam_local.available():
            ms, extra = sam_local.find_members(frame, args.concept)
        else:
            from .sam import find_members
            ms, extra = find_members(frame, args.concept), []
        img = Image.fromarray(frame).convert("RGBA")
        layer = Image.new("RGBA", img.size)
        d = ImageDraw.Draw(layer)
        for e in extra:
            d.polygon([tuple(p) for p in e["outline"]], fill=(86, 180, 233, 90))
        for m in ms:
            d.line([tuple(m.p0), tuple(m.p1)], fill=(38, 196, 170, 255), width=3)
            d.text((m.p0[0] + 6, m.p0[1] + 4), m.name, fill=(255, 255, 255, 255))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Image.alpha_composite(img, layer).convert("RGB").save(args.out)
        for m, e in zip(ms, extra or [{}] * len(ms)):
            print(f"  {m.name}: ({m.p0[0]:.0f},{m.p0[1]:.0f}) -> ({m.p1[0]:.0f},{m.p1[1]:.0f})  "
                  f"half-width {m.half_width:.0f} px" + (f"  score {e['score']:.2f}" if e else ""))
        print(f"{len(ms)} members in {time.time() - t0:.1f} s; overlay {args.out}")
        return
    if args.cmd == "check":
        from .clipcheck import check
        r = check(args.video, capture_fps=args.capture_fps)
        for c in r["checks"]:
            print(f"  {c['status']:<4}  {c['key']:<10}  {c['text']}")
        print(r["summary"])
        return
    if args.cmd == "confirm":
        from .takes import confirm
        a = json.loads((Path(args.run_a) / "report.json").read_text())
        b = json.loads((Path(args.run_b) / "report.json").read_text())
        rep = confirm(a, b)
        p = save(rep, args.out)
        for m in rep["members"]:
            ct = m.get("cross_take", {})
            print(f"{m['name']:>8}  {m['ladder']['status']:<11} other take: {ct.get('other_status')}  "
                  f"agree: {ct.get('agree')}")
        print(f"report: {p}")
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
