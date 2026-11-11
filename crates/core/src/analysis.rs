//! "Which other parts of the spectrum belong to the sound I marked?"
//!
//! The user drags a rectangle (time × frequency) around part of a sound –
//! say the fundamental of a bird call. This module looks for other
//! time-frequency regions that are likely produced by the same source, using
//! two kinds of evidence:
//!
//! 1. **Harmonic relationship with a drifting pitch.** The dominant
//!    frequency inside the rectangle is tracked frame by frame (f(t)). A
//!    voice or a bird rarely holds its pitch, so overtones are searched
//!    along r·f(t) for rational ratios r = m/n: integer multiples (the marked
//!    part is the fundamental) as well as fractions (the marked part is
//!    itself an overtone, or a sub-harmonic is present). A stationary
//!    component that happens to sit near r·f(t) for a moment does not stay
//!    on the moving track, so it is rejected.
//!
//! 2. **Common modulation.** Resonances and formants do not move with the
//!    pitch but are excited by the same source, so they swell and fade
//!    together with it. Every frequency bin whose envelope correlates
//!    strongly with the envelope of the marked part is included.
//!
//! The result is a set of *soft* masks (0..1 per time-frequency bin) so that
//! both the display and the processing can weight membership instead of
//! making hard decisions. Learning a spectral template (NMF) so the same
//! sound can be found elsewhere in the recording is planned as a third kind
//! of evidence.

use crate::noise::NoiseEstimate;
use crate::stats::{pearson, smoothstep};
use crate::stft::{Stft, StftConfig};
use serde::{Deserialize, Serialize};

/// A rectangle in the spectrogram, in physical units so it survives changes
/// of FFT size.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    /// Start time in seconds.
    pub start_s: f64,
    /// End time in seconds.
    pub end_s: f64,
    /// Lower frequency bound in Hz.
    pub low_hz: f32,
    /// Upper frequency bound in Hz.
    pub high_hz: f32,
}

impl Selection {
    /// Returns the same rectangle with start ≤ end and low ≤ high, which is
    /// what you get no matter in which direction the user dragged.
    pub fn normalized(self) -> Self {
        Self {
            start_s: self.start_s.min(self.end_s),
            end_s: self.start_s.max(self.end_s),
            low_hz: self.low_hz.min(self.high_hz),
            high_hz: self.low_hz.max(self.high_hz),
        }
    }

    /// Global frame range `[first, last)` covered by the selection.
    pub fn frames(&self, cfg: StftConfig, sample_rate: u32) -> (i64, i64) {
        let s = self.normalized();
        let to_frame = |t: f64| t * sample_rate as f64 / cfg.hop as f64;
        let first = to_frame(s.start_s).floor() as i64;
        let last = (to_frame(s.end_s).ceil() as i64).max(first + 1);
        (first, last)
    }

    /// Bin range `[low, high]` (inclusive) covered by the selection.
    pub fn bins(&self, cfg: StftConfig, sample_rate: u32) -> (usize, usize) {
        let s = self.normalized();
        let max = cfg.bins() - 1;
        let lo = (cfg.hz_to_bin(s.low_hz, sample_rate).floor().max(0.0) as usize).min(max);
        let hi = (cfg.hz_to_bin(s.high_hz, sample_rate).ceil().max(0.0) as usize).clamp(lo, max);
        (lo, hi)
    }
}

/// Tuning knobs of the correlation search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalysisParams {
    /// Largest numerator m of a harmonic ratio m/n.
    pub max_harmonic: u32,
    /// Largest denominator n of a harmonic ratio m/n (1 = only overtones).
    pub max_subharmonic: u32,
    /// Relative frequency tolerance around a predicted partial (0.03 ≈ ½
    /// semitone).
    pub harmonic_tolerance: f32,
    /// Score (0..1) above which a harmonic track counts as part of the sound.
    pub harmonic_threshold: f32,
    /// Envelope correlation above which a bin counts as co-modulated.
    pub comodulation_threshold: f32,
}

impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            max_harmonic: 12,
            max_subharmonic: 3,
            harmonic_tolerance: 0.03,
            harmonic_threshold: 0.5,
            comodulation_threshold: 0.8,
        }
    }
}

/// A harmonic partial that was found to belong to the marked sound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoundPartial {
    /// Frequency ratio to the tracked pitch (e.g. 2.0 = octave above).
    pub ratio: f32,
    /// Evidence score in 0..1.
    pub score: f32,
}

/// Result of analysing one selection: soft masks for frames
/// `first_frame .. first_frame + frames`, all bins.
#[derive(Clone, Debug, Default)]
pub struct RegionAnalysis {
    /// Global index of the first frame covered by the masks.
    pub first_frame: i64,
    /// Number of frames covered.
    pub frames: usize,
    /// Number of bins per frame.
    pub bins: usize,
    /// The energy the user marked (inside the rectangle, above the noise).
    pub selected: Vec<Vec<f32>>,
    /// Harmonically related partials found along the pitch track.
    pub harmonics: Vec<Vec<f32>>,
    /// Bins that swell and fade together with the marked part.
    pub comodulated: Vec<Vec<f32>>,
    /// Tracked dominant bin per frame (None where the sound is silent).
    pub pitch_track: Vec<Option<f32>>,
    /// Partials accepted by the harmonic search.
    pub partials: Vec<FoundPartial>,
}

impl RegionAnalysis {
    /// Mask values of one global frame, or `None` outside the analysed span.
    pub fn frame(&self, frame: i64) -> Option<(&[f32], &[f32], &[f32])> {
        let i = frame - self.first_frame;
        if i < 0 || i as usize >= self.frames {
            return None;
        }
        let i = i as usize;
        Some((&self.selected[i], &self.harmonics[i], &self.comodulated[i]))
    }
}

/// Seconds of audio around the selection used to estimate the noise floor.
const NOISE_CONTEXT_S: f64 = 2.0;

/// Analyses `selection` in the mono mix `mono`.
///
/// Runs its own STFT over the selection plus some context for the noise
/// estimate, so it can run on a background thread independently of the
/// chunk engine. Cost is proportional to the selection length only.
pub fn analyze_selection(
    mono: &[f32],
    sample_rate: u32,
    stft: &Stft,
    selection: &Selection,
    noise: &NoiseEstimate,
    params: &AnalysisParams,
) -> RegionAnalysis {
    let cfg = stft.config();
    let bins = cfg.bins();
    let (f0, f1) = selection.frames(cfg, sample_rate);
    let ctx = (NOISE_CONTEXT_S * sample_rate as f64 / cfg.hop as f64) as i64;
    let total_frames = cfg.frame_count(mono.len()) as i64;
    let ctx_start = (f0 - ctx).max(0).min(f0);
    let ctx_end = (f1 + ctx).min(total_frames).max(f1);
    let power = stft.analyze(mono, ctx_start, (ctx_end - ctx_start) as usize).power();
    let noise_power = crate::noise::NoiseReduction {
        estimate: noise.clone(),
        ..Default::default()
    }
    .noise_power(&power, bins);
    let off = (f0 - ctx_start) as usize;
    let span = &power[off..off + (f1 - f0) as usize];
    let (k0, k1) = selection.bins(cfg, sample_rate);
    analyze_power(span, f0, &noise_power, (k0, k1), params)
}

/// Core of the analysis on precomputed power frames `power` (global frame
/// index of the first one is `first_frame`), the noise power per bin and the
/// marked bin range (inclusive). Split out so it can be tested on synthetic
/// spectra.
pub fn analyze_power(
    power: &[Vec<f32>],
    first_frame: i64,
    noise: &[f32],
    (k0, k1): (usize, usize),
    params: &AnalysisParams,
) -> RegionAnalysis {
    let frames = power.len();
    let bins = noise.len();
    // Presence: how much of a bin's energy is above the noise, as a soft
    // weight (the maximum-likelihood Wiener gain). Factor 2 on the noise
    // keeps random noise peaks from looking like signal.
    let presence: Vec<Vec<f32>> = power
        .iter()
        .map(|f| {
            f.iter()
                .zip(noise)
                .map(|(&p, &n)| (1.0 - 2.0 * n / p.max(1e-20)).max(0.0))
                .collect()
        })
        .collect();
    // Magnitude weighted by presence: the "signal part" of each bin.
    let weighted: Vec<Vec<f32>> = power
        .iter()
        .zip(&presence)
        .map(|(f, pr)| f.iter().zip(pr).map(|(&p, &w)| p.sqrt() * w).collect())
        .collect();

    let mut selected = vec![vec![0.0f32; bins]; frames];
    for t in 0..frames {
        for k in k0..=k1.min(bins - 1) {
            selected[t][k] = presence[t][k];
        }
    }

    // Envelope of the marked part and the frames in which it is active.
    let envelope: Vec<f32> = weighted.iter().map(|f| f[k0..=k1.min(bins - 1)].iter().sum()).collect();
    let peak = envelope.iter().cloned().fold(0.0f32, f32::max);
    let active: Vec<usize> = (0..frames)
        .filter(|&t| peak > 0.0 && envelope[t] > 0.1 * peak)
        .collect();
    let env_active: Vec<f32> = active.iter().map(|&t| envelope[t]).collect();

    // Pitch track: dominant bin inside the rectangle, refined by parabolic
    // interpolation of the log magnitudes around the peak.
    let mut pitch_track = vec![None; frames];
    for &t in &active {
        let row = &weighted[t];
        let (mut best, mut best_v) = (k0, 0.0f32);
        for (k, &v) in row.iter().enumerate().take(k1.min(bins - 1) + 1).skip(k0) {
            if v > best_v {
                best = k;
                best_v = v;
            }
        }
        if best_v <= 0.0 {
            continue;
        }
        let mut pos = best as f32;
        if best > 0 && best + 1 < bins {
            let (a, b, c) = (
                (row[best - 1] + 1e-12).ln(),
                (best_v + 1e-12).ln(),
                (row[best + 1] + 1e-12).ln(),
            );
            let denom = a - 2.0 * b + c;
            if denom.abs() > 1e-9 {
                pos += (0.5 * (a - c) / denom).clamp(-0.5, 0.5);
            }
        }
        pitch_track[t] = Some(pos);
    }

    // If the tracked peak is mostly not above the noise, the selection holds
    // no tonal sound and every "partial" found would be a noise artefact.
    let tracked: Vec<f32> = (0..frames)
        .filter_map(|t| pitch_track[t].map(|p| presence[t][(p.round() as usize).min(bins - 1)]))
        .collect();
    let tonal = !tracked.is_empty() && tracked.iter().sum::<f32>() / tracked.len() as f32 >= 0.6;

    let mut harmonics = vec![vec![0.0f32; bins]; frames];
    let mut partials = Vec::new();
    let ratios = if tonal {
        candidate_ratios(params.max_harmonic, params.max_subharmonic)
    } else {
        Vec::new()
    };
    for ratio in ratios {
        let tol = |centre: f32| (centre * params.harmonic_tolerance).max(2.0);
        // Strength along the predicted track, and how often the track is
        // actually occupied by energy above the noise.
        let mut track = Vec::with_capacity(active.len());
        let mut occupied = 0usize;
        let mut usable = 0usize;
        for &t in &active {
            let Some(p) = pitch_track[t] else {
                track.push(0.0);
                continue;
            };
            let centre = p * ratio;
            if centre < 1.0 || centre > (bins - 2) as f32 {
                track.push(0.0);
                continue;
            }
            usable += 1;
            let (lo, hi) = band(centre, tol(centre), bins);
            let (mut v, mut pres) = (0.0f32, 0.0f32);
            for k in lo..=hi {
                v = v.max(weighted[t][k]);
                pres = pres.max(presence[t][k]);
            }
            if pres >= 0.5 {
                occupied += 1;
            }
            track.push(v);
        }
        if usable < active.len().max(1).div_ceil(2) {
            continue; // track leaves the spectrum most of the time
        }
        let coverage = occupied as f32 / usable as f32;
        let corr = pearson(&track, &env_active).max(0.0);
        // Coverage is the main evidence (the partial follows the drifting
        // pitch); a shared envelope adds confidence. A steady tone gives no
        // envelope evidence, so coverage alone can reach the threshold.
        let score = coverage * (0.6 + 0.4 * corr);
        let weight = smoothstep(params.harmonic_threshold, params.harmonic_threshold + 0.25, score);
        if weight <= 0.0 {
            continue;
        }
        partials.push(FoundPartial { ratio, score });
        for &t in &active {
            let Some(p) = pitch_track[t] else { continue };
            let centre = p * ratio;
            if centre < 1.0 || centre > (bins - 2) as f32 {
                continue;
            }
            let (lo, hi) = band(centre, tol(centre), bins);
            for k in lo..=hi {
                let m = weight * presence[t][k];
                harmonics[t][k] = harmonics[t][k].max(m);
            }
        }
    }

    // Common modulation needs a varying envelope; with a constant one every
    // correlation is just noise.
    let mut comodulated = vec![vec![0.0f32; bins]; frames];
    if active.len() >= 4 && coefficient_of_variation(&env_active) > 0.1 {
        let edge = params.comodulation_threshold;
        let mut series = Vec::with_capacity(active.len());
        for k in 0..bins {
            series.clear();
            series.extend(active.iter().map(|&t| weighted[t][k]));
            let mean_presence = active.iter().map(|&t| presence[t][k]).sum::<f32>() / active.len() as f32;
            if mean_presence < 0.3 {
                continue;
            }
            let w = smoothstep(edge, edge + (1.0 - edge) * 0.5, pearson(&series, &env_active));
            if w <= 0.0 {
                continue;
            }
            for t in 0..frames {
                comodulated[t][k] = w * presence[t][k];
            }
        }
    }

    RegionAnalysis {
        first_frame,
        frames,
        bins,
        selected,
        harmonics,
        comodulated,
        pitch_track,
        partials,
    }
}

/// All reduced fractions m/n ≠ 1 with m ≤ `max_m` and n ≤ `max_n`, sorted.
pub fn candidate_ratios(max_m: u32, max_n: u32) -> Vec<f32> {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let mut r = Vec::new();
    for n in 1..=max_n.max(1) {
        for m in 1..=max_m.max(1) {
            if m != n && gcd(m, n) == 1 {
                r.push(m as f32 / n as f32);
            }
        }
    }
    r.sort_by(f32::total_cmp);
    r
}

/// Inclusive bin range `centre ± tol`, clipped to the spectrum.
fn band(centre: f32, tol: f32, bins: usize) -> (usize, usize) {
    let lo = (centre - tol).floor().max(0.0) as usize;
    let hi = ((centre + tol).ceil() as usize).min(bins - 1);
    (lo, hi)
}

/// Standard deviation divided by the mean.
fn coefficient_of_variation(v: &[f32]) -> f32 {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    var.sqrt() / mean
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::Stft;

    /// Synthesises a "voice": a pitch gliding from 300 to 450 Hz with
    /// harmonics 2 and 3, an unrelated steady tone at 1700 Hz, and noise.
    fn gliding_voice(sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        let mut rng = 12345u64;
        let mut phase = 0.0f32;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let f0 = 300.0 + 150.0 * t / secs;
                phase += 2.0 * std::f32::consts::PI * f0 / sr as f32;
                // amplitude swells and fades so envelopes carry information
                let env = (std::f32::consts::PI * t / secs).sin();
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.002;
                env * (0.5 * phase.sin() + 0.25 * (2.0 * phase).sin() + 0.15 * (3.0 * phase).sin())
                    + 0.1 * (2.0 * std::f32::consts::PI * 1700.0 * t).sin()
                    + noise
            })
            .collect()
    }

    #[test]
    fn finds_harmonics_of_a_gliding_fundamental_but_not_unrelated_tone() {
        let sr = 16000;
        let x = gliding_voice(sr, 2.0);
        let cfg = StftConfig {
            fft_size: 1024,
            hop: 256,
        };
        let stft = Stft::new(cfg);
        // Mark the fundamental only.
        let sel = Selection {
            start_s: 0.3,
            end_s: 1.7,
            low_hz: 280.0,
            high_hz: 470.0,
        };
        let a = analyze_selection(
            &x,
            sr,
            &stft,
            &sel,
            &NoiseEstimate::default(),
            &AnalysisParams::default(),
        );
        let ratios: Vec<f32> = a.partials.iter().map(|p| p.ratio).collect();
        assert!(ratios.contains(&2.0), "octave not found: {ratios:?}");
        assert!(ratios.contains(&3.0), "fifth above octave not found: {ratios:?}");

        // In the middle of the selection the 2nd harmonic bin is marked and
        // the steady 1700 Hz tone is not.
        let mid = (a.frames / 2) as i64 + a.first_frame;
        let (_, harm, comod) = a.frame(mid).unwrap();
        let t_rel = (mid as f32 * cfg.hop as f32) / sr as f32;
        let f0 = 300.0 + 150.0 * t_rel / 2.0;
        let k2 = cfg.hz_to_bin(2.0 * f0, sr).round() as usize;
        assert!(harm[k2] > 0.5, "2nd harmonic weight {}", harm[k2]);
        let k_tone = cfg.hz_to_bin(1700.0, sr).round() as usize;
        assert!(harm[k_tone] < 0.1 && comod[k_tone] < 0.1, "steady tone was marked");
    }

    #[test]
    fn noise_only_selection_finds_no_partials() {
        let sr = 16000;
        let x = gliding_voice(sr, 2.0);
        let stft = Stft::new(StftConfig {
            fft_size: 1024,
            hop: 256,
        });
        // 5–6 kHz contains only the white noise.
        let sel = Selection {
            start_s: 0.3,
            end_s: 1.7,
            low_hz: 5000.0,
            high_hz: 6000.0,
        };
        let a = analyze_selection(
            &x,
            sr,
            &stft,
            &sel,
            &NoiseEstimate::default(),
            &AnalysisParams::default(),
        );
        assert!(a.partials.is_empty(), "found {:?} in noise", a.partials);
    }

    #[test]
    fn selected_mask_stays_inside_rectangle() {
        let sr = 16000;
        let x = gliding_voice(sr, 1.0);
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let sel = Selection {
            start_s: 0.2,
            end_s: 0.8,
            low_hz: 250.0,
            high_hz: 500.0,
        };
        let a = analyze_selection(
            &x,
            sr,
            &stft,
            &sel,
            &NoiseEstimate::default(),
            &AnalysisParams::default(),
        );
        let (k0, k1) = sel.bins(stft.config(), sr);
        for row in &a.selected {
            for (k, &v) in row.iter().enumerate() {
                if k < k0 || k > k1 {
                    assert_eq!(v, 0.0);
                }
            }
        }
        assert!(a.selected.iter().flatten().any(|&v| v > 0.9));
    }

    #[test]
    fn comodulated_resonance_is_found() {
        // A fixed resonance at bin 40 follows the envelope of the marked
        // band at bin 10; bin 70 has a different envelope.
        let bins = 100;
        let noise = vec![1.0; bins];
        let power: Vec<Vec<f32>> = (0..60)
            .map(|t| {
                let env = 1.0 + 500.0 * (std::f32::consts::PI * t as f32 / 60.0).sin().powi(2);
                let other = 1.0 + 500.0 * (t as f32 / 60.0);
                let mut f = vec![1.0; bins];
                f[10] = env;
                f[40] = env * 0.3;
                f[70] = other;
                f
            })
            .collect();
        let a = analyze_power(&power, 0, &noise, (8, 12), &AnalysisParams::default());
        assert!(a.comodulated[30][40] > 0.5);
        assert!(a.comodulated[30][70] < 0.1);
    }

    #[test]
    fn candidate_ratios_are_reduced_and_exclude_unity() {
        let r = candidate_ratios(4, 2);
        assert_eq!(r, vec![0.5, 1.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn selection_normalizes_drag_direction() {
        let s = Selection {
            start_s: 2.0,
            end_s: 1.0,
            low_hz: 900.0,
            high_hz: 100.0,
        }
        .normalized();
        assert_eq!((s.start_s, s.end_s, s.low_hz, s.high_hz), (1.0, 2.0, 100.0, 900.0));
        let cfg = StftConfig {
            fft_size: 1000,
            hop: 100,
        };
        assert_eq!(s.frames(cfg, 1000), (10, 20));
        assert_eq!(s.bins(cfg, 1000), (100, 500));
    }
}
