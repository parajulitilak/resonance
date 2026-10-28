"""Resonance on real footage with independent ground truth (licensed clips in the research folder)."""
import sys

sys.path.insert(0, "src")
from resonance.analyze import analyze  # noqa: E402
from resonance.members import patch  # noqa: E402

F = "/Volumes/Data/PERSONAL/Applications/2026-10 Meta Global AI Developer Hackathon/research/footage"

# 1. Cello G string, 7000 fps (Mores & Plath 2018, CC BY 4.0): 6.15 g/m, 0.68 m, wave speed 133 m/s -> T = 108.8 N
rep = analyze(f"{F}/cello_string_7000fps/Torsion6_crf8.mp4",
              [patch("string_point", (100, 200, 196, 280), direction=(1, 0), length_m=0.68, mass_per_m=0.00615,
                     tension_range=(50, 200))], capture_fps=7000)
m = rep["members"][0]
T = m["tension"]["value_n"]
print(f"cello: {m['ladder']['status']}, f0 {m['ladder']['f0_hz']:.2f} Hz, {len(m['ladder']['modes'])} rungs, "
      f"T {T:.1f} ± {m['tension']['sigma_n']:.1f} N vs 108.8 N ({(T - 108.8) / 108.8:+.1%})")

# 2. Oak tree sway, 30 fps (Ammatelli et al. 2023, CC BY 4.0): accelerometer 0.23 Hz (60 s), 0.24 Hz (30 min)
rep = analyze(f"{F}/treesway_trout_lake/trout-e.MP4",
              [patch("trunk_x", (900, 520, 1156, 904), direction=(1, 0)),
               patch("trunk_y", (900, 520, 1156, 904), direction=(0, 1))])
for m in rep["members"]:
    print(f"tree {m['name']}: {m['ladder']['f0_hz']:.3f} Hz vs accelerometer 0.23-0.24 Hz")
