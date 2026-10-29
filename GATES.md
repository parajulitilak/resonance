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
