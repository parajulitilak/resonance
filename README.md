# Resonance

Cable tension and vibration from a phone video. SAM 3.1 places the sensors, physics checks itself, and a Muse Spark agent writes the report an engineer signs off.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). Everything in this repository was written during the event.

## How it works

1. **Timing.** Frame times come from the container's per-frame timestamps, never the nominal frame rate, so slow-motion files with drifting or relabelled rates still give true frequencies.
2. **Sensors.** Each cable is a line of stations. At every station the cross profile is sampled and the cable's position is tracked to a fraction of a pixel. Camera shake is measured on a static anchor (phase correlation, upsampled DFT) and removed along each cable's normal.
3. **Spectrum.** Displacement from all stations is combined into one spectrum; peaks are found against a running noise floor.
4. **Self-check.** A taut cable rings at `f_n = n f0 sqrt(1 + B n^2)`. Resonance only reports a number when the strong peaks sit on one such ladder; aliased or ambiguous takes are refused with a reason.
5. **Physics.** Taut string with pinned ends: `T = 4 m L^2 f0^2`, with propagated uncertainty. Members whose bending stiffness is significant (`xi < 10`) get relative comparisons only.

## Status

| Piece | State |
|---|---|
| Timing, tracking, shake removal, spectrum, harmonic verifier, tension | working; synthetic bench: 5/5 ropes within 0.02% at 240 fps, all refused at 30 fps |
| SAM 3.1 member detection | next |
| Muse Spark agent and Muse Spark Open verifier | next |
| Field mode (sag, neighbour comparison) | next |

## Try it

```bash
export PYTHONPATH=src
python -m resonance.cli synth media/bench240.mp4 --fps 240
python -m resonance.cli analyze media/bench240.mp4 --truth media/bench240.mp4.json
python -m pytest -q
```

Requires Python 3.12, numpy, scipy, Pillow and ffmpeg.
