//! Where a sound comes from, as seen by a multi-channel recording.
//!
//! A source at a fixed position reaches the microphones with a fixed
//! *level difference* (it is closer to, or aimed at, one of them) and a
//! fixed *delay* (the sound travels further to one of them). Both are
//! properties of the place, not of the sound, so they help to decide
//! whether a repetition of a call comes from the same bird or from its
//! neighbour singing the same song somewhere else.
//!
//! For every channel `c ≥ 1` the signature holds the level difference to
//! channel 0 in dB and the delay of channel `c` relative to channel 0 in
//! samples (positive: the sound arrives later at channel `c`). Only the
//! time-frequency cells that belong to the sound are used, so background
//! noise from other directions does not blur the estimate.
//!
//! The delay is estimated with GCC-PHAT (generalised cross-correlation with
//! phase transform): the cross-spectrum of the two channels is normalised
//! to unit magnitude per cell, so every cell votes with its phase only, and
//! the delay whose phase ramp agrees best with those votes wins. Delays are
//! searched in quarter-sample steps; with spaced microphones they reach tens
//! of samples, with coincident microphones they are near zero and the level
//! difference carries the information.

use crate::stft::Stft;
use realfft::num_complex::Complex32;

/// Level and delay of one channel relative to channel 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairCue {
    /// Level of the channel minus level of channel 0, in dB.
    pub level_db: f32,
    /// Arrival delay at the channel relative to channel 0, in samples.
    pub delay_samples: f32,
}

/// Spatial signature of a sound: one [`PairCue`] per channel after the
/// first. Empty for mono recordings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Signature {
    /// Cues of channels 1, 2, ... relative to channel 0.
    pub pairs: Vec<PairCue>,
}

/// Cells of one sound to measure: `frames` STFT frames starting at global
/// frame `first_frame`, bins `k0 .. k0 + weights.len() / frames`, each with
/// a membership weight (frame-major).
pub struct Cells<'a> {
    /// Global frame of the first row.
    pub first_frame: i64,
    /// Number of frames.
    pub frames: usize,
    /// First bin.
    pub k0: usize,
    /// Membership weights, `frames × width`, frame-major.
    pub weights: &'a [f32],
}

/// Step of the delay search in samples.
const DELAY_STEP: f32 = 0.25;

/// Per-bin sums over the frames of some [`Cells`] for one channel pair.
struct PairSums {
    /// Σ w·cross/|cross| per bin (phase votes, cross = ref·conj(other)).
    votes: Vec<Complex32>,
    /// Σ w·|ref|² per bin.
    e_ref: Vec<f64>,
    /// Σ w·|other|² per bin.
    e_other: Vec<f64>,
}

/// Analyses all channels over the frames of `cells` and sums, for every
/// channel after the first, the membership-weighted cues per bin.
fn pair_sums(channels: &[Vec<f32>], stft: &Stft, cells: &Cells) -> Vec<PairSums> {
    let width = cells.weights.len() / cells.frames.max(1);
    let spectra: Vec<Vec<Vec<Complex32>>> = channels
        .iter()
        .map(|ch| stft.analyze(ch, cells.first_frame, cells.frames).frames)
        .collect();
    let reference = &spectra[0];
    spectra[1..]
        .iter()
        .map(|other| {
            let mut sums = PairSums {
                votes: vec![Complex32::new(0.0, 0.0); width],
                e_ref: vec![0.0; width],
                e_other: vec![0.0; width],
            };
            for t in 0..cells.frames {
                for i in 0..width {
                    let w = cells.weights[t * width + i];
                    if w <= 0.0 {
                        continue;
                    }
                    let k = cells.k0 + i;
                    let (a, b) = (reference[t][k], other[t][k]);
                    sums.e_ref[i] += (w * a.norm_sqr()) as f64;
                    sums.e_other[i] += (w * b.norm_sqr()) as f64;
                    let cross = a * b.conj();
                    let mag = cross.norm();
                    if mag > 1e-20 {
                        sums.votes[i] += cross * (w / mag);
                    }
                }
            }
            sums
        })
        .collect()
}

/// Phase of bin `k` that a delay of `tau` samples puts on the votes
/// ref·conj(other): a delay d of the other channel turns its spectrum into
/// X·e^{-j2πkd/N}, so the votes rotate by e^{+j2πkd/N}.
fn delay_phase(k: usize, tau: f32, fft_size: f32) -> f32 {
    2.0 * std::f32::consts::PI * k as f32 * tau / fft_size
}

/// Measures the spatial signature of `cells` in `channels`.
///
/// Returns `None` for fewer than two channels or when the cells carry no
/// weight or no energy.
pub fn measure(channels: &[Vec<f32>], stft: &Stft, cells: &Cells, max_delay_samples: f32) -> Option<Signature> {
    if channels.len() < 2 || cells.frames == 0 {
        return None;
    }
    let width = cells.weights.len() / cells.frames;
    if width == 0 || cells.weights.iter().all(|&w| w <= 0.0) {
        return None;
    }
    let n = stft.config().fft_size as f32;
    let mut pairs = Vec::with_capacity(channels.len() - 1);
    for sums in pair_sums(channels, stft, cells) {
        let e_ref: f64 = sums.e_ref.iter().sum();
        let e_other: f64 = sums.e_other.iter().sum();
        if e_ref <= 0.0 || e_other <= 0.0 {
            return None;
        }
        let level_db = (10.0 * (e_other / e_ref).log10()) as f32;
        // Undoing the rotation of a candidate delay τ aligns the votes best
        // at the true delay.
        let score = |tau: f32| -> f32 {
            sums.votes
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let phase = -delay_phase(cells.k0 + i, tau, n);
                    v.re * phase.cos() - v.im * phase.sin()
                })
                .sum()
        };
        let steps = (max_delay_samples.max(0.0) / DELAY_STEP).round() as i32;
        let (mut best_tau, mut best) = (0.0f32, f32::MIN);
        for s in -steps..=steps {
            let tau = s as f32 * DELAY_STEP;
            let v = score(tau);
            // Prefer the smaller delay on ties (symmetric phase wrapping).
            if v > best + 1e-6 || ((v - best).abs() <= 1e-6 && tau.abs() < best_tau.abs()) {
                best = v;
                best_tau = tau;
            }
        }
        pairs.push(PairCue {
            level_db,
            delay_samples: best_tau,
        });
    }
    Some(Signature { pairs })
}

/// How well each bin of `cells` agrees with the direction `reference`
/// (0..1, 1 for bins without weight or energy).
///
/// Per bin and channel pair, the level difference is compared directly.
/// A single bin only knows its delay modulo N/k samples, so the delay
/// error is the phase error to the reference delay's phase, converted to
/// samples: `Δ = wrap(φ)·N/(2πk)`. At high frequencies that error can
/// never exceed the tolerance, which is right: there the phase cannot tell
/// the directions apart and the level difference has to.
pub fn per_bin_agreement(
    channels: &[Vec<f32>],
    stft: &Stft,
    cells: &Cells,
    reference: &Signature,
    level_tolerance_db: f32,
    delay_tolerance_samples: f32,
) -> Vec<f32> {
    let width = cells.weights.len() / cells.frames.max(1);
    let mut out = vec![1.0f32; width];
    if channels.len() < 2 || cells.frames == 0 {
        return out;
    }
    let n = stft.config().fft_size as f32;
    let level_tol = level_tolerance_db.max(1e-3);
    let delay_tol = delay_tolerance_samples.max(1e-3);
    for (sums, cue) in pair_sums(channels, stft, cells).iter().zip(&reference.pairs) {
        for i in 0..width {
            let (er, eo) = (sums.e_ref[i], sums.e_other[i]);
            if er <= 0.0 || eo <= 0.0 {
                continue;
            }
            let k = cells.k0 + i;
            let dl = ((10.0 * (eo / er).log10()) as f32 - cue.level_db) / level_tol;
            let v = sums.votes[i];
            let dd = if k == 0 || v.norm() <= 1e-20 {
                0.0
            } else {
                let aligned = v * Complex32::from_polar(1.0, -delay_phase(k, cue.delay_samples, n));
                aligned.arg() * n / (2.0 * std::f32::consts::PI * k as f32) / delay_tol
            };
            out[i] *= (-0.5 * (dl * dl + dd * dd)).exp();
        }
    }
    out
}

/// How well two signatures agree, 1 for identical, falling off like a
/// Gaussian with the given tolerances (product over channel pairs).
pub fn agreement(a: &Signature, b: &Signature, level_tolerance_db: f32, delay_tolerance_samples: f32) -> f32 {
    a.pairs
        .iter()
        .zip(&b.pairs)
        .map(|(x, y)| {
            let dl = (x.level_db - y.level_db) / level_tolerance_db.max(1e-3);
            let dd = (x.delay_samples - y.delay_samples) / delay_tolerance_samples.max(1e-3);
            (-0.5 * (dl * dl + dd * dd)).exp()
        })
        .product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::StftConfig;

    /// Band-limited noise burst (a sum of sines) placed in two channels with
    /// the given gain (dB) and delay (whole samples) on the second channel.
    fn stereo(gain_db: f32, delay: usize) -> Vec<Vec<f32>> {
        let n = 8000;
        let src: Vec<f32> = (0..n)
            .map(|i| {
                (1..40)
                    .map(|h| ((i as f32) * 0.013 * h as f32 + h as f32 * 1.7).sin() / h as f32)
                    .sum::<f32>()
            })
            .collect();
        let g = 10f32.powf(gain_db / 20.0);
        let right: Vec<f32> = (0..n)
            .map(|i| if i >= delay { g * src[i - delay] } else { 0.0 })
            .collect();
        vec![src, right]
    }

    /// All cells of frames 10..30, bins 2..200 with weight 1.
    fn measure_all(ch: &[Vec<f32>]) -> Signature {
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let weights = vec![1.0; 20 * 198];
        let cells = Cells {
            first_frame: 10,
            frames: 20,
            k0: 2,
            weights: &weights,
        };
        measure(ch, &stft, &cells, 32.0).unwrap()
    }

    #[test]
    fn measures_level_and_delay() {
        let s = measure_all(&stereo(-6.0, 7));
        assert_eq!(s.pairs.len(), 1);
        assert!((s.pairs[0].level_db + 6.0).abs() < 0.5, "{s:?}");
        assert!((s.pairs[0].delay_samples - 7.0).abs() <= 0.25, "{s:?}");
        let s = measure_all(&stereo(3.0, 0));
        assert!((s.pairs[0].level_db - 3.0).abs() < 0.5, "{s:?}");
        assert!(s.pairs[0].delay_samples.abs() <= 0.25, "{s:?}");
    }

    #[test]
    fn mono_or_empty_cells_give_no_signature() {
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let weights = vec![1.0; 10];
        let cells = Cells {
            first_frame: 0,
            frames: 1,
            k0: 0,
            weights: &weights,
        };
        assert!(measure(&[vec![0.0; 1000]], &stft, &cells, 8.0).is_none());
        let zero = vec![0.0; 10];
        let cells = Cells {
            weights: &zero,
            ..cells
        };
        assert!(measure(&stereo(0.0, 0), &stft, &cells, 8.0).is_none());
    }

    #[test]
    fn agreement_falls_with_distance() {
        let a = Signature {
            pairs: vec![PairCue {
                level_db: 0.0,
                delay_samples: 0.0,
            }],
        };
        let near = Signature {
            pairs: vec![PairCue {
                level_db: 1.0,
                delay_samples: 0.5,
            }],
        };
        let far = Signature {
            pairs: vec![PairCue {
                level_db: -9.0,
                delay_samples: 12.0,
            }],
        };
        assert_eq!(agreement(&a, &a, 3.0, 2.0), 1.0);
        assert!(agreement(&a, &near, 3.0, 2.0) > 0.8);
        assert!(agreement(&a, &far, 3.0, 2.0) < 0.01);
    }

    #[test]
    fn per_bin_agreement_separates_directions() {
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let reference = measure_all(&stereo(-6.0, 7));
        let weights = vec![1.0; 20 * 198];
        let cells = Cells {
            first_frame: 10,
            frames: 20,
            k0: 2,
            weights: &weights,
        };
        let mean = |ch: &[Vec<f32>]| {
            let a = per_bin_agreement(ch, &stft, &cells, &reference, 3.0, 2.0);
            a.iter().sum::<f32>() / a.len() as f32
        };
        assert!(mean(&stereo(-6.0, 7)) > 0.9);
        // Another level is seen in every bin.
        assert!(mean(&stereo(6.0, 0)) < 0.05);
        // Another delay only: a bin k knows the delay modulo N/k samples,
        // so a 13-sample error is only visible below k = N/26 ≈ 19; above,
        // phase wrapping makes some bins agree by accident.
        let low = vec![1.0; 20 * 16];
        let cells = Cells { weights: &low, ..cells };
        let a = per_bin_agreement(&stereo(-6.0, 20), &stft, &cells, &reference, 3.0, 2.0);
        assert!(a.iter().sum::<f32>() / (a.len() as f32) < 0.1, "{a:?}");
        // Mono: nothing to judge, everything agrees.
        assert!(
            per_bin_agreement(&[vec![0.0; 8000]], &stft, &cells, &reference, 3.0, 2.0)
                .iter()
                .all(|&a| a == 1.0)
        );
    }
}
