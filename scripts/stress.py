"""Robustness matrix on the synthetic rig: shake x compression x noise."""
import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
from resonance import synth  # noqa: E402
from resonance.analyze import analyze, compare_with_truth, members_from_truth  # noqa: E402

out = Path("out/stress")
out.mkdir(parents=True, exist_ok=True)
rows = []
for shake, crf, noise in itertools.product([0.6, 3.0], [16, 28], [2.0, 6.0]):
    vid = out / f"s{shake}_c{crf}_n{noise}.mp4"
    truth = synth.render(vid, fps=240, duration=4.0, shake_px=shake, crf=crf, noise=noise, seed=7)
    for anchor in (True, False):
        rep = analyze(str(vid), members_from_truth(truth), anchor=tuple(truth["anchor_box"]) if anchor else None)
        cmp = compare_with_truth(rep, truth)
        errs = [abs(r["err"]) for r in cmp if r["err"] is not None]
        rows.append({"shake_px": shake, "crf": crf, "noise": noise, "anchor": anchor,
                     "ok": sum(1 for r in cmp if r["ladder_ok"]), "n": len(cmp),
                     "max_err": max(errs) if errs else None})
        print(json.dumps(rows[-1]), flush=True)
    vid.unlink()
    Path(str(vid) + ".json").unlink()
(out / "stress.json").write_text(json.dumps(rows, indent=2))
