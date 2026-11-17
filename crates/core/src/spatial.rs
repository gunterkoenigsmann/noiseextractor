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
use serde::{Deserialize, Serialize};

/// Level and delay of one channel relative to channel 0.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairCue {
    /// Level of the channel minus level of channel 0, in dB.
    pub level_db: f32,
    /// Arrival delay at the channel relative to channel 0, in samples.
    pub delay_samples: f32,
    /// How far (in samples, either way) the true delay may be from
    /// `delay_samples` as far as the measured cells can tell: a narrow band
    /// pins the delay down only loosely, and in a room the reflections
    /// shift each bin's phase a little (see [`DELAY_PIN_FRACTION`]).
    #[serde(default)]
    pub delay_uncertainty: f32,
}

/// The delays whose GCC-PHAT score is at least this fraction of the best
/// one are taken as equally plausible (see [`PairCue::delay_uncertainty`]).
/// 0.9 corresponds to a mean phase misfit of about 25°, which reflections
/// in an ordinary room easily cause; a single narrow band at bin k then
/// leaves about ±N/(14·k) samples open, a broadband sound far less.
pub const DELAY_PIN_FRACTION: f32 = 0.9;

/// Spatial signature of a sound: one [`PairCue`] per channel after the
/// first. Empty for mono recordings. Saved with automatically discovered
/// sounds (see [`crate::discover`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    /// Noise power per bin (all bins) of each channel, if known. A faint
    /// sound's cells hold a good deal of background noise, which in a
    /// diffuse field is about equally loud in every channel and has random
    /// phase; it pulls the measured level difference towards 0 dB and
    /// blurs the delay, so a quiet repetition from the same place looks as
    /// if it came from elsewhere (issue #22). With the noise known, each
    /// cell is weighted by how far it stands above the noise and the noise
    /// power is taken out of the level sums (see [`PairSums::add_above_noise`]).
    pub noise: Option<&'a [Vec<f32>]>,
}

/// Step of the delay search in samples.
const DELAY_STEP: f32 = 0.25;

/// Per-bin sums over the frames of some cells for one channel pair: what
/// a [`Signature`] is estimated from (see [`from_sums`]).
#[derive(Clone, Debug)]
pub struct PairSums {
    /// Σ w·cross/|cross| per bin (phase votes, cross = ref·conj(other)).
    votes: Vec<Complex32>,
    /// Σ w·|ref|² per bin.
    e_ref: Vec<f64>,
    /// Σ w·|other|² per bin.
    e_other: Vec<f64>,
}

impl PairSums {
    /// Empty sums for `width` bins.
    pub fn new(width: usize) -> Self {
        Self {
            votes: vec![Complex32::new(0.0, 0.0); width],
            e_ref: vec![0.0; width],
            e_other: vec![0.0; width],
        }
    }

    /// Adds one cell of bin index `i` with membership weight `w`, whose
    /// spectrum is `a` in the reference channel and `b` in the other one.
    pub fn add(&mut self, i: usize, w: f32, a: Complex32, b: Complex32) {
        self.add_above_noise(i, w, a, b, (0.0, 0.0));
    }

    /// Like [`PairSums::add`] for a cell whose channels carry the noise
    /// powers `(n_a, n_b)`.
    ///
    /// Noise adds its power to both channels on average, so subtracting it
    /// from the level sums leaves the source's own level ratio. That is
    /// only unbiased if the level weight does not depend on the cell's own
    /// (noisy) power, so the level sums keep the plain membership weight
    /// `w`. The delay votes are unit phasors, which noise does not bias but
    /// scatters; there the weight is scaled by the cell's share above the
    /// noise, `1 − (n_a + n_b) / (|a|² + |b|²)`, so noise-only cells
    /// hardly vote.
    pub fn add_above_noise(&mut self, i: usize, w: f32, a: Complex32, b: Complex32, (n_a, n_b): (f32, f32)) {
        if w <= 0.0 {
            return;
        }
        let (p_a, p_b) = (a.norm_sqr(), b.norm_sqr());
        self.e_ref[i] += (w * (p_a - n_a)) as f64;
        self.e_other[i] += (w * (p_b - n_b)) as f64;
        let vote = if n_a + n_b > 0.0 {
            w * (1.0 - (n_a + n_b) / (p_a + p_b).max(1e-30)).max(0.0)
        } else {
            w
        };
        let cross = a * b.conj();
        let mag = cross.norm();
        if mag > 1e-20 {
            self.votes[i] += cross * (vote / mag);
        }
    }
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
        .enumerate()
        .map(|(c, other)| {
            // Index of this channel in `channels` (and in `cells.noise`).
            let c = c + 1;
            let mut sums = PairSums::new(width);
            for t in 0..cells.frames {
                for i in 0..width {
                    let k = cells.k0 + i;
                    let noise = cells.noise.map_or((0.0, 0.0), |n| (n[0][k], n[c][k]));
                    sums.add_above_noise(i, cells.weights[t * width + i], reference[t][k], other[t][k], noise);
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
    from_sums(
        &pair_sums(channels, stft, cells),
        cells.k0,
        stft.config().fft_size,
        max_delay_samples,
    )
}

/// Estimates a signature from per-bin sums, one [`PairSums`] per channel
/// after the first, whose bin index 0 is bin `k0` of an `fft_size`-point
/// STFT. Delays are searched up to ±`max_delay_samples`. `None` when a
/// channel pair carries no energy.
pub fn from_sums(sums: &[PairSums], k0: usize, fft_size: usize, max_delay_samples: f32) -> Option<Signature> {
    if sums.is_empty() {
        return None;
    }
    let n = fft_size as f32;
    let mut pairs = Vec::with_capacity(sums.len());
    for sums in sums {
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
                    let phase = -delay_phase(k0 + i, tau, n);
                    v.re * phase.cos() - v.im * phase.sin()
                })
                .sum()
        };
        let steps = (max_delay_samples.max(0.0) / DELAY_STEP).round() as i32;
        let scores: Vec<f32> = (-steps..=steps).map(|s| score(s as f32 * DELAY_STEP)).collect();
        let (mut best_i, mut best) = (0usize, f32::MIN);
        for (i, &v) in scores.iter().enumerate() {
            let tau = (i as i32 - steps) as f32 * DELAY_STEP;
            let best_tau = (best_i as i32 - steps) as f32 * DELAY_STEP;
            // Prefer the smaller delay on ties (symmetric phase wrapping).
            if v > best + 1e-6 || ((v - best).abs() <= 1e-6 && tau.abs() < best_tau.abs()) {
                best = v;
                best_i = i;
            }
        }
        // The plateau around the best delay: as far as the score stays
        // within DELAY_PIN_FRACTION of the best, on each side.
        let pinned = |v: f32| best > 0.0 && v >= DELAY_PIN_FRACTION * best;
        let left = scores[..best_i].iter().rev().take_while(|&&v| pinned(v)).count();
        let right = scores[best_i + 1..].iter().take_while(|&&v| pinned(v)).count();
        pairs.push(PairCue {
            level_db,
            delay_samples: (best_i as i32 - steps) as f32 * DELAY_STEP,
            delay_uncertainty: left.max(right) as f32 * DELAY_STEP,
        });
    }
    Some(Signature { pairs })
}

/// How far one bin's cues are from a reference direction, for one channel
/// pair (see [`per_bin_deviation`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Deviation {
    /// Level difference minus the reference's, in dB.
    pub level_db: f32,
    /// Delay error in samples beyond the reference's own uncertainty, as
    /// far as the bin's phase can tell (see [`per_bin_deviation`]).
    pub delay_samples: f32,
    /// Membership-weighted energy of the bin in channel 0; 0 for bins
    /// without weight or energy, whose deviations mean nothing.
    pub weight: f32,
}

/// Deviation of each bin of `cells` from the direction `reference`, one
/// vector (indexed like the bins of `cells`) per channel pair.
///
/// Per bin and channel pair, the level difference is compared directly.
/// A single bin only knows its delay modulo N/k samples, so the delay
/// error is the phase error to the reference delay's phase, converted to
/// samples: `Δ = wrap(φ)·N/(2πk)`. At high frequencies that error can
/// never exceed the tolerance, which is right: there the phase cannot tell
/// the directions apart and the level difference has to. Errors within the
/// reference's [`PairCue::delay_uncertainty`] count as none, since the
/// reference itself may be off by that much.
pub fn per_bin_deviation(
    channels: &[Vec<f32>],
    stft: &Stft,
    cells: &Cells,
    reference: &Signature,
) -> Vec<Vec<Deviation>> {
    let width = cells.weights.len() / cells.frames.max(1);
    if channels.len() < 2 || cells.frames == 0 {
        return Vec::new();
    }
    let n = stft.config().fft_size as f32;
    pair_sums(channels, stft, cells)
        .iter()
        .zip(&reference.pairs)
        .map(|(sums, cue)| {
            (0..width)
                .map(|i| {
                    let (er, eo) = (sums.e_ref[i], sums.e_other[i]);
                    if er <= 0.0 || eo <= 0.0 {
                        return Deviation::default();
                    }
                    let k = cells.k0 + i;
                    let v = sums.votes[i];
                    let delay_samples = if k == 0 || v.norm() <= 1e-20 {
                        0.0
                    } else {
                        let aligned = v * Complex32::from_polar(1.0, -delay_phase(k, cue.delay_samples, n));
                        let err = aligned.arg() * n / (2.0 * std::f32::consts::PI * k as f32);
                        err.signum() * (err.abs() - cue.delay_uncertainty).max(0.0)
                    };
                    Deviation {
                        level_db: (10.0 * (eo / er).log10()) as f32 - cue.level_db,
                        delay_samples,
                        weight: er as f32,
                    }
                })
                .collect()
        })
        .collect()
}

/// Energy-weighted RMS of the level and delay deviations over all bins and
/// channel pairs: how much the cues of one sound scatter around its own
/// direction. Reflections in a room make every bin see a slightly
/// different level difference and phase, so this is the yardstick for how
/// far a bin may be off and still come from the same place. `(0, 0)` when
/// nothing carries weight.
pub fn spread(deviations: &[Vec<Deviation>]) -> (f32, f32) {
    let (mut w, mut l, mut d) = (0.0f64, 0.0f64, 0.0f64);
    for dev in deviations.iter().flatten() {
        let dw = dev.weight as f64;
        w += dw;
        l += dw * (dev.level_db as f64).powi(2);
        d += dw * (dev.delay_samples as f64).powi(2);
    }
    if w <= 0.0 {
        return (0.0, 0.0);
    }
    ((l / w).sqrt() as f32, (d / w).sqrt() as f32)
}

/// Turns deviations into an agreement per bin (0..1, 1 for bins without
/// weight or energy): a Gaussian in level and delay error with the given
/// tolerances, multiplied over the channel pairs.
pub fn agreement_of(
    deviations: &[Vec<Deviation>],
    width: usize,
    level_tolerance_db: f32,
    delay_tolerance_samples: f32,
) -> Vec<f32> {
    let mut out = vec![1.0f32; width];
    let level_tol = level_tolerance_db.max(1e-3);
    let delay_tol = delay_tolerance_samples.max(1e-3);
    for pair in deviations {
        for (o, dev) in out.iter_mut().zip(pair) {
            if dev.weight <= 0.0 {
                continue;
            }
            let (dl, dd) = (dev.level_db / level_tol, dev.delay_samples / delay_tol);
            *o *= (-0.5 * (dl * dl + dd * dd)).exp();
        }
    }
    out
}

/// How well each bin of `cells` agrees with the direction `reference`
/// (0..1, 1 for bins without weight or energy): [`per_bin_deviation`]
/// judged by [`agreement_of`] with fixed tolerances.
pub fn per_bin_agreement(
    channels: &[Vec<f32>],
    stft: &Stft,
    cells: &Cells,
    reference: &Signature,
    level_tolerance_db: f32,
    delay_tolerance_samples: f32,
) -> Vec<f32> {
    let width = cells.weights.len() / cells.frames.max(1);
    agreement_of(
        &per_bin_deviation(channels, stft, cells, reference),
        width,
        level_tolerance_db,
        delay_tolerance_samples,
    )
}

/// How well two signatures agree, 1 for identical, falling off like a
/// Gaussian with the given tolerances (product over channel pairs).
///
/// A delay difference within what the two measurements can pin down
/// (their [`PairCue::delay_uncertainty`], combined in quadrature) counts as
/// none, as in [`per_bin_deviation`]: a faint or narrow-band sound leaves
/// its delay open by several samples, and a delay picked anywhere on that
/// plateau says nothing about a different place.
pub fn agreement(a: &Signature, b: &Signature, level_tolerance_db: f32, delay_tolerance_samples: f32) -> f32 {
    a.pairs
        .iter()
        .zip(&b.pairs)
        .map(|(x, y)| {
            let dl = (x.level_db - y.level_db) / level_tolerance_db.max(1e-3);
            let open = x.delay_uncertainty.hypot(y.delay_uncertainty);
            let dd = ((x.delay_samples - y.delay_samples).abs() - open).max(0.0) / delay_tolerance_samples.max(1e-3);
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
            noise: None,
            weights: &weights,
        };
        measure(ch, &stft, &cells, 32.0).unwrap()
    }

    /// Issue #22: the same stereo source (−7 dB, 3 samples), once loud and
    /// once 20 dB quieter in diffuse noise (independent, equally loud in
    /// both channels). Without knowing the noise, the quiet one's level
    /// difference is pulled towards 0 dB and it looks as if it came from
    /// elsewhere; with the noise known it measures like the loud one.
    #[test]
    fn faint_source_in_diffuse_noise_keeps_its_direction() {
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let mut rng = 4242u64;
        let mut white = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let noise: Vec<Vec<f32>> = (0..2).map(|_| (0..8000).map(|_| 0.2 * white()).collect()).collect();
        let mix = |gain: f32| -> Vec<Vec<f32>> {
            stereo(-7.0, 3)
                .iter()
                .zip(&noise)
                .map(|(s, n)| s.iter().zip(n).map(|(s, n)| gain * s + n).collect())
                .collect()
        };
        // Noise power per bin and channel, as a template would know it.
        let noise_power: Vec<Vec<f32>> = noise
            .iter()
            .map(|ch| crate::noise::mean_power(&stft.analyze(ch, 0, 60).power(), 257))
            .collect();
        // The cells the source occupies (its partials reach bin ~42), as a
        // template's membership would select them.
        let weights = vec![1.0; 20 * 40];
        let cells = |noise| Cells {
            first_frame: 10,
            frames: 20,
            k0: 2,
            noise,
            weights: &weights,
        };
        let known = Some(noise_power.as_slice());
        let loud = measure(&mix(1.0), &stft, &cells(known), 32.0).unwrap();
        let faint_blind = measure(&mix(0.1), &stft, &cells(None), 32.0).unwrap();
        let faint = measure(&mix(0.1), &stft, &cells(known), 32.0).unwrap();
        assert!((loud.pairs[0].level_db + 7.0).abs() < 0.5, "{loud:?}");
        // The noise does pull the blind measurement towards 0 dB ...
        assert!(faint_blind.pairs[0].level_db > -5.5, "{faint_blind:?}");
        // ... and knowing it undoes that.
        assert!((faint.pairs[0].level_db + 7.0).abs() < 1.0, "{faint:?}");
        let same = |s: &Signature| agreement(&loud, s, 3.0, 2.0);
        assert!(same(&faint) > 0.9, "same place {}", same(&faint));
        assert!(same(&faint) > same(&faint_blind) + 0.1);
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
            noise: None,
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
                delay_uncertainty: 0.0,
            }],
        };
        let near = Signature {
            pairs: vec![PairCue {
                level_db: 1.0,
                delay_samples: 0.5,
                delay_uncertainty: 0.0,
            }],
        };
        let far = Signature {
            pairs: vec![PairCue {
                level_db: -9.0,
                delay_samples: 12.0,
                delay_uncertainty: 0.0,
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
            noise: None,
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
