"""Resonance on real footage with independent ground truth (licensed clips in the research folder)."""
import os
import sys

sys.path.insert(0, "src")
from resonance.analyze import analyze  # noqa: E402
from resonance.members import patch  # noqa: E402

F = os.environ.get("RESONANCE_FOOTAGE", "data/footage")  # folder with the downloaded clips

# 1. Cello G string, 7000 fps (Mores & Plath 2018, CC BY 4.0): 6.15 g/m, 0.68 m, wave speed 133 m/s -> T = 108.8 N
# Two takes of the same string (Torsion6 high bow force, Torsion3 low bow force) test repeatability.
for take in ("Torsion6", "Torsion3"):
    rep = analyze(f"{F}/cello_string_7000fps/{take}_crf8.mp4",
                  [patch("string_point", (100, 200, 196, 280), direction=(1, 0), length_m=0.68, mass_per_m=0.00615,
                         tension_range=(50, 200))], capture_fps=7000)
    m = rep["members"][0]
    T = m["tension"]["value_n"]
    print(f"cello {take}: {m['ladder']['status']}, f0 {m['ladder']['f0_hz']:.2f} Hz, {len(m['ladder']['modes'])} rungs, "
          f"T {T:.1f} ± {m['tension']['sigma_n']:.1f} N vs 108.8 N ({(T - 108.8) / 108.8:+.1%})")

# 2. Oak tree sway, 30 fps (Ammatelli et al. 2023, CC BY 4.0): accelerometer 0.23 Hz (60 s), 0.24 Hz (30 min)
rep = analyze(f"{F}/treesway_trout_lake/trout-e.MP4",
              [patch("trunk_x", (900, 520, 1156, 904), direction=(1, 0)),
               patch("trunk_y", (900, 520, 1156, 904), direction=(0, 1))])
for m in rep["members"]:
    print(f"tree {m['name']}: {m['ladder']['f0_hz']:.3f} Hz vs accelerometer 0.23-0.24 Hz")

# 3. Stock footage that is NOT measurement-grade: Resonance must refuse, with the reason
for name, v, ms, anchor in [
        ("forest footbridge, runners crossing", f"{F}/pexels_forest_bridge_runners_side/pexels_forest_bridge_runners_side.mp4",
         [patch("deck", (500, 410, 660, 490))], None),
        ("red footbridge, drifting camera", f"{F}/pexels_red_suspension_bridge_crowd_60fps/pexels_red_suspension_bridge_crowd_60fps.mp4",
         [patch("deck", (1120, 490, 1280, 550))], (80, 660, 200, 840))]:
    m = analyze(v, ms, anchor=anchor)["members"][0]
    print(f"{name}: {m['ladder']['status']} ({m['ladder']['reasons'][0][:70]}…)")
