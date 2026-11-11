//! Stationary background noise: estimating its spectrum and removing it with
//! a soft, decision-directed spectral gain.
//!
//! ## Why not a gate?
//!
//! Zeroing every bin that is quieter than the expected noise level changes
//! the shape of every sound: its quiet attack and decay tails fall below the
//! threshold and vanish, and inside a sound the weaker partials vanish while
//! the strong ones stay, so the timbre changes. The estimators here instead
//! compute a gain per bin from the *a-priori SNR* ξ, smoothed over time with
//! Ephraim & Malah's decision-directed rule:
//!
//! ```text
//! γ(t,k) = P(t,k) / N(k)                          a-posteriori SNR
//! ξ(t,k) = α·G(t-1,k)²·γ(t-1,k) + (1-α)·max(γ(t,k)-1, 0)
//! ```
//!
//! Where a sound is clearly above the noise, ξ is large and the gain is ~1,
//! so the sound passes nearly unchanged. Where only noise is present the
//! gain sinks towards the floor; it never reaches zero, which keeps the
//! residual noise natural instead of "musical".

use crate::stats::{expint_e1, quantile};
use serde::{Deserialize, Serialize};

/// Which gain rule turns the SNR estimates into a mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GainRule {
    /// Wiener gain ξ/(1+ξ): simple, slightly more residual noise.
    Wiener,
    /// MMSE log-spectral amplitude estimator (Ephraim & Malah 1985): usually
    /// sounds most natural, with little musical noise.
    #[default]
    MmseLsa,
}

/// How the noise spectrum is obtained.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NoiseEstimate {
    /// Estimated automatically from the quietest frames around each chunk.
    /// `quantile` is the fraction of frames assumed to contain only noise.
    Adaptive {
        /// Fraction (0..1) of the quietest frames treated as noise.
        quantile: f32,
    },
    /// A fixed noise power spectrum (one value per bin), for example learned
    /// from a selection that contains only background noise.
    Profile(Vec<f32>),
}

impl Default for NoiseEstimate {
    fn default() -> Self {
        NoiseEstimate::Adaptive { quantile: 0.1 }
    }
}

/// Parameters of the background-noise reduction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseReduction {
    /// Where the noise spectrum comes from.
    pub estimate: NoiseEstimate,
    /// Gain rule.
    pub rule: GainRule,
    /// Maximum attenuation in dB; the gain never drops below this.
    pub reduction_db: f32,
    /// Decision-directed smoothing factor α (0.9 – 0.99 typical). Higher is
    /// smoother but reacts slower to onsets.
    pub smoothing: f32,
    /// Multiplier applied to the noise estimate. Values > 1 remove more.
    pub over_subtraction: f32,
}

impl Default for NoiseReduction {
    fn default() -> Self {
        Self {
            estimate: NoiseEstimate::default(),
            rule: GainRule::default(),
            reduction_db: 18.0,
            smoothing: 0.98,
            over_subtraction: 1.0,
        }
    }
}

/// Lower bound for ξ (−25 dB) as recommended by Cappé (1994); prevents the
/// gain from collapsing in noise-only frames, which causes musical noise.
const XI_MIN: f32 = 0.003_162;

impl NoiseReduction {
    /// Gain floor as a linear amplitude factor.
    pub fn gain_floor(&self) -> f32 {
        10f32.powf(-self.reduction_db.max(0.0) / 20.0)
    }

    /// Noise power per bin to use for the given frames: either the fixed
    /// profile or an adaptive estimate from `power` itself.
    pub fn noise_power(&self, power: &[Vec<f32>], bins: usize) -> Vec<f32> {
        let mut n = match &self.estimate {
            NoiseEstimate::Adaptive { quantile } => estimate_noise(power, bins, *quantile),
            NoiseEstimate::Profile(p) if p.len() == bins => p.clone(),
            // A profile from another FFT size cannot be used bin by bin;
            // resample it linearly rather than failing.
            NoiseEstimate::Profile(p) => resample(p, bins),
        };
        for v in &mut n {
            *v *= self.over_subtraction.max(0.0);
        }
        n
    }

    /// Computes the gain for every frame and bin of `power`, given the noise
    /// power `noise`. The first frames act as warm-up for the recursive ξ
    /// estimate; callers that process chunks start a few frames early.
    pub fn gains(&self, power: &[Vec<f32>], noise: &[f32]) -> Vec<Vec<f32>> {
        let alpha = self.smoothing.clamp(0.0, 0.999);
        let floor = self.gain_floor();
        let bins = noise.len();
        // |Â(t-1)|² / N, i.e. G(t-1)²·γ(t-1); starts at "no speech".
        let mut prev_clean_snr = vec![0.0f32; bins];
        let mut out = Vec::with_capacity(power.len());
        for frame in power {
            let mut g = vec![1.0f32; bins];
            for k in 0..bins.min(frame.len()) {
                let n = noise[k].max(1e-20);
                let gamma = (frame[k] / n).min(1e6);
                let xi = (alpha * prev_clean_snr[k] + (1.0 - alpha) * (gamma - 1.0).max(0.0)).max(XI_MIN);
                let gain = match self.rule {
                    GainRule::Wiener => xi / (1.0 + xi),
                    GainRule::MmseLsa => lsa_gain(xi, gamma),
                };
                prev_clean_snr[k] = gain * gain * gamma;
                g[k] = gain.max(floor).min(1.0);
            }
            out.push(g);
        }
        out
    }
}

/// MMSE-LSA gain G = ξ/(1+ξ) · exp(½·E1(v)), v = ξγ/(1+ξ).
fn lsa_gain(xi: f32, gamma: f32) -> f32 {
    let wiener = xi / (1.0 + xi);
    let v = (wiener * gamma).max(1e-8) as f64;
    // For large v the correction factor tends to 1; E1 underflows anyway.
    let corr = if v > 30.0 {
        1.0
    } else {
        (0.5 * expint_e1(v)).exp() as f32
    };
    (wiener * corr).min(1.0)
}

/// Estimates the noise power spectrum from frames that contain noise most of
/// the time: per bin, the `q`-quantile of the power over time, corrected for
/// the fact that a low quantile of noise power underestimates its mean.
///
/// For Gaussian noise the power in a bin is exponentially distributed, whose
/// q-quantile is `-ln(1-q)` times its mean; dividing by that factor yields an
/// unbiased mean estimate when the quietest `q` of frames are noise-only.
pub fn estimate_noise(power: &[Vec<f32>], bins: usize, q: f32) -> Vec<f32> {
    let q = q.clamp(0.01, 0.9);
    let correction = -(1.0 - q).ln();
    let mut column = Vec::with_capacity(power.len());
    (0..bins)
        .map(|k| {
            column.clear();
            column.extend(power.iter().filter_map(|f| f.get(k).copied()));
            quantile(&column, q) / correction
        })
        .collect()
}

/// Mean power per bin; used to learn a noise profile from a selection the
/// user marked as "noise only".
pub fn mean_power(power: &[Vec<f32>], bins: usize) -> Vec<f32> {
    let mut acc = vec![0.0f64; bins];
    for f in power {
        for (a, &p) in acc.iter_mut().zip(f) {
            *a += p as f64;
        }
    }
    let n = power.len().max(1) as f64;
    acc.into_iter().map(|a| (a / n) as f32).collect()
}

/// Linear resampling of a spectrum to a different number of bins.
fn resample(src: &[f32], bins: usize) -> Vec<f32> {
    if src.is_empty() || bins == 0 {
        return vec![0.0; bins];
    }
    (0..bins)
        .map(|k| {
            let pos = k as f32 * (src.len() - 1) as f32 / (bins.max(2) - 1) as f32;
            let i = pos.floor() as usize;
            let frac = pos - i as f32;
            let a = src[i.min(src.len() - 1)];
            let b = src[(i + 1).min(src.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random generator so tests are reproducible.
    pub(crate) struct Lcg(pub u64);
    impl Lcg {
        /// Uniform value in [-1, 1).
        pub fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// Exponentially distributed power values with the given mean, as the
    /// power of complex Gaussian noise would be.
    fn noise_frames(frames: usize, bins: usize, mean: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg(seed);
        (0..frames)
            .map(|_| {
                (0..bins)
                    .map(|_| -mean * (1.0 - (rng.next() * 0.5 + 0.5)).max(1e-9).ln())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn adaptive_estimate_finds_noise_mean() {
        let p = noise_frames(2000, 4, 2.0, 7);
        for v in estimate_noise(&p, 4, 0.1) {
            // within ±1 dB
            assert!((10.0 * (v / 2.0).log10()).abs() < 1.0, "estimate {v}");
        }
    }

    #[test]
    fn adaptive_estimate_ignores_loud_minority() {
        let mut p = noise_frames(1000, 2, 1.0, 3);
        for f in p.iter_mut().take(400) {
            f[0] *= 1000.0; // 40 % of frames contain a loud sound
        }
        let n = estimate_noise(&p, 2, 0.1);
        assert!(n[0] < 2.0, "loud frames leaked into the noise estimate: {}", n[0]);
    }

    #[test]
    fn gains_pass_loud_sounds_and_attenuate_noise() {
        let nr = NoiseReduction::default();
        let noise = vec![1.0; 2];
        let mut power = noise_frames(200, 2, 1.0, 11);
        for f in power.iter_mut().skip(100) {
            f[1] = 1000.0; // 30 dB above the noise in bin 1 from frame 100
        }
        let g = nr.gains(&power, &noise);
        let floor = nr.gain_floor();
        // noise-only bin ends up near the floor
        let mean_noise_gain: f32 = g[50..100].iter().map(|f| f[0]).sum::<f32>() / 50.0;
        assert!(mean_noise_gain < 3.0 * floor, "noise gain {mean_noise_gain}");
        // the loud sound passes almost unattenuated a few frames after onset
        assert!(g[105][1] > 0.95, "signal gain {}", g[105][1]);
        assert!(g.iter().flatten().all(|&x| x >= floor - 1e-6 && x <= 1.0));
    }

    #[test]
    fn loud_sound_keeps_its_relative_amplitudes() {
        // Two partials 12 dB apart, both well above the noise: a soft mask
        // must keep their ratio (a gate at the noise level would too, but
        // here we also check the decaying tail below).
        let nr = NoiseReduction {
            rule: GainRule::Wiener,
            ..Default::default()
        };
        let noise = vec![1.0; 2];
        let power: Vec<Vec<f32>> = (0..60)
            .map(|t| {
                let env = 10000.0 * (-(t as f32) / 15.0).exp(); // decaying sound
                vec![env, env / 16.0]
            })
            .collect();
        let g = nr.gains(&power, &noise);
        for t in 5..25 {
            let ratio_in = power[t][0] / power[t][1];
            let ratio_out = (power[t][0] * g[t][0] * g[t][0]) / (power[t][1] * g[t][1] * g[t][1]);
            let change_db = 10.0 * (ratio_out / ratio_in).log10();
            assert!(
                change_db.abs() < 1.5,
                "frame {t}: partial ratio changed by {change_db} dB"
            );
        }
        // The gain falls smoothly with the decay instead of jumping to the
        // floor at a threshold.
        for t in 1..60 {
            assert!(g[t][0] <= g[t - 1][0] + 0.05);
        }
    }

    #[test]
    fn lsa_gain_is_between_wiener_and_one_for_high_snr() {
        let g = lsa_gain(10.0, 11.0);
        assert!(g > 10.0 / 11.0 && g <= 1.0);
        assert!(lsa_gain(0.003, 0.5) < 0.2);
    }

    #[test]
    fn profile_is_resampled_when_bin_count_differs() {
        let nr = NoiseReduction {
            estimate: NoiseEstimate::Profile(vec![0.0, 1.0, 2.0]),
            ..Default::default()
        };
        let n = nr.noise_power(&[], 5);
        assert_eq!(n, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn mean_power_averages_frames() {
        assert_eq!(mean_power(&[vec![1.0, 2.0], vec![3.0, 4.0]], 2), vec![2.0, 3.0]);
    }
}
