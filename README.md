# Resonance

Cable tension and vibration from a phone video. SAM 3.1 places the sensors, physics checks itself, and a Muse Spark agent writes the report an engineer signs off.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). Everything in this repository was written during the event.

## How it works

1. **Timing.** Frame times come from the container's per-frame timestamps, never the nominal frame rate, so slow-motion files with drifting or relabelled rates still give true frequencies.
2. **Sensors.** Each cable is a line of stations. At every station the cross profile is sampled and the cable's position is tracked to a fraction of a pixel. Camera shake is measured on a static anchor (phase correlation, upsampled DFT) and removed along each cable's normal.
3. **Spectrum.** Displacement from all stations is combined into one spectrum; peaks are found against a running noise floor.
4. **Self-checks.** A taut cable rings at `f_n = n f0 sqrt(1 + B n^2)`. A result is *verified* only if the strong peaks sit on one such ladder, the fundamental's mode shape has no interior node (and the 2nd has one), the frame rate is high enough for a coarse frequency prior (the alias guard), and the frame timing is steady. Otherwise it is *provisional* (labelled, with a re-shoot request) or *refused* with the reason.
5. **Physics.** Taut string with pinned ends: `T = 4 m L^2 f0^2`, with propagated uncertainty and a pinned/clamped bracket when end fixity matters. Members whose bending stiffness is significant (`xi < 10`) get relative comparisons only.
6. **Agent.** A Muse Spark inspector writes the report from tool calls; code enforces the rules (no numbers on refused members, provisional labelled, no "safe", no absolute field tension); Muse Spark Open independently verifies every claim and fails closed; an engineer signs off, bound to a hash of the evidence.
7. **Field mode.** Main-cable sag from the fitted curvature and the sway frequency `sqrt(g / 8b)` are two independent readings of the same cable.

## Status

| Piece | State |
|---|---|
| Timing, tracking, shake removal, spectrum, verifiers, tension | working; synthetic bench: 5/5 ropes within 0.02% at 240 fps, all refused at 30 fps |
| Inspection agent + independent verifier + approval | working on Muse Spark Open (Baseten) |
| Field mode (sag, sway frequency) | working on a synthetic footbridge |
| SAM 3.1 member detection | client written against the documented wire format; not yet run against the live API |
| Real phone footage | not yet tested |

Known limits: shake removal is translation-only (camera-coincident peaks are removed); sag assumes a roughly fronto-parallel view.

## Try it

```bash
export PYTHONPATH=src
python -m resonance.cli synth media/bench240.mp4 --fps 240
python -m resonance.cli analyze media/bench240.mp4 --truth media/bench240.mp4.json
python -m pytest -q
```

Requires Python 3.12, numpy, scipy, Pillow and ffmpeg.
