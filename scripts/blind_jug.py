"""Synthetic blind test: re-film after someone lightens one jug. Does Resonance name the rope?"""
import json
import sys

sys.path.insert(0, "src")
from resonance import synth  # noqa: E402
from resonance.analyze import analyze, members_from_truth, save  # noqa: E402
from resonance.fingerprint import describe, diff  # noqa: E402

ropes = synth.default_rig()
ropes[2].mass_kg = 4.5   # rope3: 6.0 kg → 4.5 kg, i.e. tension -25%
t = synth.render("media/bench240_recheck.mp4", fps=240, duration=5.0, ropes=ropes, seed=11)
rep = analyze("media/bench240_recheck.mp4", members_from_truth(t), anchor=tuple(t["anchor_box"]))
save(rep, "out/bench240_recheck")
base = json.load(open("out/bench240/report.json"))
res = diff(base, rep)
for r in res["members"]:
    print(("⚠ " if r["status"] == "changed" else "  ") + describe(r))
print("truth: rope3 tension -25.0%, all others unchanged")
