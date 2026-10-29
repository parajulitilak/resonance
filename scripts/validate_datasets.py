"""Resonance on open datasets with independent references (research/datasets, downloaded 2026-10-29)."""
import sys

import numpy as np

sys.path.insert(0, "src")
from resonance.analyze import analyze  # noqa: E402
from resonance.members import Member, patch  # noqa: E402

D = "/Volumes/Data/PERSONAL/Applications/2026-10 Meta Global AI Developer Hackathon/research/datasets"

# 1. FLAMBRG bridge-cable model in a wind tunnel, 30 fps side view (Trush et al., Zenodo 15655246, CC BY 4.0).
#    References from the accelerometers: free decay 5.07/10.04/15.01 Hz; wind runs 9.55 Hz (5.0 m/s), 9.57 (6.3),
#    18.35 (10.2), 19.2/20.9 (11.3). At 30 fps anything above 15 Hz aliases.
print("== FLAMBRG bridge-cable model (30 fps)")
for k in (1, 2, 3, 4):
    v = f"{D}/zenodo15655246_flambrg_cable_windtunnel/iced90_5Hz_yaw0_mode{k}_side.mp4"
    # prior from the record: low pretension tuned to f1 = 5 Hz (±20%)
    m = Member(f"cable_mode{k}", (300, 451), (1700, 427), half_width=40, f0_range=(4.0, 6.0))
    e = analyze(v, [m])["members"][0]
    L = e["ladder"]
    print(f"  clip 'mode {k}': {L['status']:<11} f0 {L['f0_hz']:.3f} Hz, rungs {[x['n'] for x in L['modes']]}, "
          f"top peaks {[round(p['f'], 2) for p in sorted(e['peaks'], key=lambda p: -p['power'])[:3]]} "
          f"| {L['reasons'][:1]}")

# 2. MIT guitar, 600 fps (EVM SIGGRAPH 2012): low E = 82.41 Hz, A = 110 Hz. Parallel line trackers across the strings.
print("== MIT guitar (600 fps)")
v = f"{D}/mit_riesz_phase_evm_strings/guitar_riesz.avi"
ms = [Member(f"line_y{y0}", (110, y0), (215, y0 + 0.19 * 105), half_width=2.5, f0_range=(60, 130))
      for y0 in range(18, 112, 3)]
rep = analyze(v, ms, capture_fps=600)
found = {}
for e in rep["members"]:
    L = e["ladder"]
    if L["status"] != "refused":
        found[e["name"]] = (L["status"], round(L["f0_hz"], 2), [x["n"] for x in L["modes"]])
for k, val in found.items():
    print(f"  {k}: {val}")
print(f"  ({len(found)} of {len(ms)} trackers verified a ladder; refs 82.41 Hz and 110 Hz)")
# frequency-level check (0.5 s clip, trackers see part of each string: no ladder, so the tool itself refuses)
from resonance.displacement import track  # noqa: E402
from resonance.spectrum import spectrum  # noqa: E402
tr = track(v, ms, capture_fps=600)
best = []
for mm in ms:
    sp = spectrum(tr.t, tr.disp[mm.name], pad=8)
    band = (sp.f > 60) & (sp.f < 130)
    i = int(np.argmax(np.where(band, sp.power, 0)))
    best.append((float(sp.power[i] / np.median(sp.power[band])), mm.name, float(sp.f[i])))
for ratio, name, f in sorted(best, reverse=True)[:4]:
    print(f"  strongest in-band peak {name}: {f:.1f} Hz (prominence {ratio:.0f}x)")

# 3. Shaking-table structure free decay, 62 fps, laser reference 4.8467 Hz (Mendeley njhcj2kphr, CC BY 4.0).
print("== Shaking table free decay (62 fps)")
v = f"{D}/mendeley_njhcj2kphr_tcvrre_freedecay/012自振2中.avi"
rep = analyze(v, [patch("table_x", (633, 300, 933, 550), direction=(1, 0)),
                  patch("table_y", (633, 300, 933, 550), direction=(0, 1))])
for e in rep["members"]:
    print(f"  {e['name']}: {e['ladder']['status']} {e['ladder']['f0_hz']:.4f} Hz vs laser 4.8467 Hz "
          f"| {e['ladder']['reasons'][:1]}")
