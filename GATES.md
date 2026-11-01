# Measurement gates (frozen as `gates-v1`)

Thresholds are frozen in the tagged commit `gates-v1` so that later accuracy and refusal results are not tuned after the fact. Change them only in a new tagged version, and report which version produced each result.

| Gate | Where | Threshold (gates-v1) |
|---|---|---|
| Harmonic ladder: rungs | `spectrum.fit_harmonics` | ≥ 3 rungs for verified; exact 2:1 pair (≤ 0.3%, SNR ≥ 20) → provisional |
| Ladder residual | same | RMS relative residual ≤ 0.6% |
| Explained power | same | ladder carries ≥ 75% of strong-peak power |
| Rung tolerance | same | ±2.5% of the (stretched) rung frequency |
| Alias guard | same | f0 inside the prior range and prior max ≤ fs / 2.5; else provisional (no prior) or refused |
| Frame-rate floor | same | 2nd harmonic below 0.92 × Nyquist |
| Peak detection | `spectrum.peaks` | prominence ≥ 6 dB, SNR ≥ 8 vs running median floor |
| Mode shape | `analyze` | rung 1 has 0 interior dips, rung 2 has 1 (amplitude dips below 0.35 × neighbouring lobe) |
| Timing | `timing.frame_times` | ramp ratio > 1.3 → refused; quantised timestamps de-quantised by linear fit |
| Camera motion | `analyze` | anchor shake RMS > 5 px → refused |
| Patch occlusion | `analyze` | frame-to-frame match, 10th percentile < 0.30 → refused (calibrated on 4 clips: provisional) |
| End fixity | `physics.tension` | ξ < 10 → relative only; pinned/clamped spread > 5% → bracket reported, not reliable |
| Change detection | `fingerprint.diff` | changed if \|ΔT/T\| > max(2%, 3σ) |

## gates-v2 (2026-10-29, before any benchmark collection)

Added after testing on the open wind-tunnel cable dataset (Zenodo 15655246), which shows single-mode wind lock-in:

| Gate | Where | Threshold (gates-v2) |
|---|---|---|
| Single-mode identification | `analyze` | only when the ladder is refused and a prior exists: dominant peak SNR ≥ 50 and ≥ 2× the next peak; mode number n = interior dips + 1 (n ≤ 6); f/n inside the prior; n × prior max ≤ fs / 2.5 (alias guard per mode). Result is provisional, never verified. |

## gates-v3 (2026-11-01)

Added after a review of the refusal path found that, without a marked static anchor, camera motion was never gated (a walking camera could pass for cable vibration). Re-validated on every clip in the README: no result changed; the moving-camera stock clips are now refused for the camera itself.

| Gate | Where | Threshold (gates-v3) |
|---|---|---|
| Camera motion without an anchor | `analyze.picture_motion` → `clipcheck.global_shake` | camera motion RMS > 5 px → refused. Motion is the median shift over textured tiles (4×3 grid at 320 px width, tile std ≥ 3 grey levels, match ≥ 0.15) against a keyframe that is renewed when fewer than a third of the tiles still match; the members being measured are left out. Measured: cello close-up 0.03 px, tree 1.4, wind tunnel 0.02, bench 0.27, drifting handheld 11, walking camera 90. |
| Clip check (advice only, before marking) | `clipcheck.check` | whole-picture motion ≤ 1 px ok, ≤ 30 px warn (the cable may fill the view; the measurement re-checks), > 30 px fail |
| Visual review of SAM candidates | `review` | Muse Spark Open verdict; a rejected candidate is not measured unless the user keeps it; any failure → "unreviewed", never "confirmed" |
