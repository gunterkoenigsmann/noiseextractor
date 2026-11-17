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
//! making hard decisions. The union of these masks also yields a
//! [`Template`](crate::similar::Template) for finding the same sound elsewhere
//! in the recording (see [`crate::similar`]).

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
#[serde(default)]
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
    /// Multi-channel recordings: level difference between channels (dB) at
    /// which a found partial or resonance whose channel ratio differs from
    /// the marked part's has its weight reduced to e^-½.
    pub spatial_level_tolerance_db: f32,
    /// Multi-channel recordings: the same for the inter-channel delay, in
    /// samples.
    pub spatial_delay_tolerance_samples: f32,
    /// Largest inter-channel delay searched, in samples (64 ≈ 1.3 ms at
    /// 48 kHz, i.e. microphones up to ~45 cm apart).
    pub max_delay_samples: f32,
}

impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            max_harmonic: 12,
            max_subharmonic: 3,
            harmonic_tolerance: 0.03,
            harmonic_threshold: 0.5,
            comodulation_threshold: 0.8,
            spatial_level_tolerance_db: 3.0,
            spatial_delay_tolerance_samples: 2.0,
            max_delay_samples: 64.0,
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
    /// Set when the marked sound is not tonal but a band of noise (insect
    /// pulses, rustling, a hiss): the lowest and highest bin of the band,
    /// i.e. of the marked bins that are mostly above the noise while the
    /// sound is active. The pitch track, partials and source model are
    /// then empty, because a "pitch" would only be the loudest bin of the
    /// noise band, jumping around from frame to frame.
    pub broadband: Option<(usize, usize)>,
    /// Spectro-temporal pattern of the whole sound, for finding it
    /// elsewhere (`None` if nothing above the noise was marked).
    pub template: Option<crate::similar::Template>,
    /// How loud the parts are relative to the fundamental, for sharing
    /// bins with other filters (`None` without a pitch track).
    pub model: Option<crate::share::SourceModel>,
}

impl RegionAnalysis {
    /// Union of all parts (selected, harmonics, co-modulated) per cell:
    /// everything the analysis attributes to the sound.
    pub fn members(&self) -> Vec<Vec<f32>> {
        (0..self.frames)
            .map(|t| {
                (0..self.bins)
                    .map(|k| {
                        self.selected[t][k]
                            .max(self.harmonics[t][k])
                            .max(self.comodulated[t][k])
                    })
                    .collect()
            })
            .collect()
    }

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
///
/// With two or more `channels` (the individual tracks of the recording;
/// pass `&[]` for mono), the marked part's direction is measured (level
/// ratio and delay between the channels), harmonics and co-modulated bins
/// coming from another direction lose weight (see [`apply_direction`]),
/// and the template remembers the direction so that repetitions can be
/// checked for it, too.
pub fn analyze_selection(
    mono: &[f32],
    channels: &[Vec<f32>],
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
    let mut analysis = analyze_power(span, f0, &noise_power, (k0, k1), params);
    if channels.len() >= 2 && apply_direction(&mut analysis, channels, stft, (k0, k1), params) {
        // The membership changed, so the template has to follow.
        analysis.template = crate::similar::Template::build(span, f0, &noise_power, &analysis.members());
        analysis.model = crate::share::SourceModel::build(span, &noise_power, &analysis, params.harmonic_tolerance);
    }
    if let Some(template) = analysis.template.as_mut() {
        let channel_noise = if channels.len() >= 2 {
            channel_noise(channels, stft, ctx_start, (ctx_end - ctx_start) as usize, noise)
        } else {
            Vec::new()
        };
        template.attach_spatial(channels, stft, params.max_delay_samples, channel_noise);
    }
    analysis
}

/// Noise power per bin of each of `channels`, over `frames` frames from
/// global frame `first`. A noise profile describes the mono mix only, so
/// each channel gets its own adaptive estimate (with the profile's
/// estimator's quantile when `noise` is adaptive, else the default one).
fn channel_noise(
    channels: &[Vec<f32>],
    stft: &Stft,
    first: i64,
    frames: usize,
    noise: &NoiseEstimate,
) -> Vec<Vec<f32>> {
    let adaptive = match noise {
        NoiseEstimate::Adaptive { .. } => noise.clone(),
        NoiseEstimate::Profile(_) => NoiseEstimate::default(),
    };
    let NoiseEstimate::Adaptive { quantile } = adaptive else {
        unreachable!("the default noise estimate is adaptive")
    };
    let bins = stft.config().bins();
    channels
        .iter()
        .map(|ch| crate::noise::estimate_noise(&stft.analyze(ch, first, frames).power(), bins, quantile))
        .collect()
}

/// How many RMS deviations of the marked part's own bins a found bin may
/// be off its direction before it loses weight like a bin one tolerance
/// off (see [`apply_direction`]). At 2, a bin that scatters like a typical
/// one of the marked part keeps about 80 % of its weight.
pub const SPREAD_FACTOR: f32 = 2.0;

/// Uses the direction of the marked part (inside the rectangle) to judge
/// what the analysis found outside of it: a bird's overtones reach the
/// microphones with the same level ratio and delay as its fundamental,
/// while a neighbour's tone that happens to sit on a harmonic or to swell
/// at the same time comes from elsewhere. Each bin of the harmonic and the
/// co-modulation masks is multiplied by its agreement with the marked
/// part's direction (see [`crate::spatial::agreement_of`]).
///
/// The tolerances grow with the scatter of the marked part's own bins
/// around its direction ([`SPREAD_FACTOR`] times their RMS deviation):
/// in a room, reflections give every bin a slightly different level
/// difference and phase, and a check stricter than the sound's agreement
/// with itself would reject its own overtones.
///
/// Returns `false` (and changes nothing) when no direction could be
/// measured.
pub fn apply_direction(
    analysis: &mut RegionAnalysis,
    channels: &[Vec<f32>],
    stft: &Stft,
    (k0, k1): (usize, usize),
    params: &AnalysisParams,
) -> bool {
    use crate::spatial::{agreement_of, measure, per_bin_deviation, spread, Cells};
    let (frames, bins) = (analysis.frames, analysis.bins);
    if frames == 0 || bins == 0 {
        return false;
    }
    let k1 = k1.min(bins - 1);
    let marked: Vec<f32> = analysis
        .selected
        .iter()
        .flat_map(|row| row[k0..=k1].iter().copied())
        .collect();
    let cells = Cells {
        first_frame: analysis.first_frame,
        frames,
        k0,
        noise: None,
        weights: &marked,
    };
    let Some(reference) = measure(channels, stft, &cells, params.max_delay_samples) else {
        return false;
    };
    let (level_spread, delay_spread) = spread(&per_bin_deviation(channels, stft, &cells, &reference));
    let level_tol = params.spatial_level_tolerance_db.max(SPREAD_FACTOR * level_spread);
    let delay_tol = params.spatial_delay_tolerance_samples.max(SPREAD_FACTOR * delay_spread);
    let gate = |mask: &mut [Vec<f32>]| {
        let weights: Vec<f32> = mask.iter().flatten().copied().collect();
        let cells = Cells {
            first_frame: analysis.first_frame,
            frames,
            k0: 0,
            noise: None,
            weights: &weights,
        };
        let agree = agreement_of(
            &per_bin_deviation(channels, stft, &cells, &reference),
            bins,
            level_tol,
            delay_tol,
        );
        for row in mask.iter_mut() {
            row.iter_mut().zip(&agree).for_each(|(m, a)| *m *= a);
        }
    };
    gate(&mut analysis.harmonics);
    gate(&mut analysis.comodulated);
    true
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

    // A noise band has no pitch: its loudest bin is random. Following it
    // would report "partials" that are just other parts of the band
    // (issue #24), so such a selection is treated as the band itself.
    let kk1 = k1.min(bins - 1);
    let broadband = is_broadband(&pitch_track, &active, &weighted, (k0, kk1), params.harmonic_tolerance)
        .then(|| noise_band(&weighted, &active, (k0, kk1)))
        .flatten();
    if broadband.is_some() {
        pitch_track.iter_mut().for_each(|p| *p = None);
    }

    // If the tracked peak is mostly not above the noise, the selection holds
    // no tonal sound and every "partial" found would be a noise artefact.
    let tracked: Vec<f32> = (0..frames)
        .filter_map(|t| pitch_track[t].map(|p| presence[t][(p.round() as usize).min(bins - 1)]))
        .collect();
    let tonal = !tracked.is_empty() && tracked.iter().sum::<f32>() / tracked.len() as f32 >= 0.6;

    // How far the envelope's shape can be trusted as evidence: 0 for a flat
    // envelope (coefficient of variation below 5 %, i.e. only noise
    // ripple), 1 for one that clearly swells and fades (above 20 %).
    let envelope_trust = smoothstep(0.05, 0.2, coefficient_of_variation(&env_active));

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
        // pitch); a shared envelope adds confidence. How much a missing
        // correlation may count against a partial depends on whether the
        // envelope carries information at all: a steady tone's envelope is
        // flat apart from noise, so its correlation is meaningless and must
        // not hold the score down (issue #16: the octave of a steady tone
        // only got a mask of ~0.35, so suppressing it left the octave).
        let score = coverage * (1.0 - envelope_trust * 0.4 * (1.0 - corr));
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

    let mut analysis = RegionAnalysis {
        first_frame,
        frames,
        bins,
        selected,
        harmonics,
        comodulated,
        pitch_track,
        partials,
        broadband,
        template: None,
        model: None,
    };
    // Template from everything the analysis attributes to the sound.
    analysis.template = crate::similar::Template::build(power, first_frame, noise, &analysis.members());
    analysis.model = crate::share::SourceModel::build(power, noise, &analysis, params.harmonic_tolerance);
    analysis
}

/// Share of the marked band's energy (above the noise) that the tracked
/// peak must stay below for a sound to count as a noise band: a tone, even
/// a weak one with overtones in the selection, puts far more of the band's
/// energy into its peak (the owl calls: ~0.7, cricket pulses: ~0.15).
const BROADBAND_MAX_PEAK_SHARE: f32 = 0.3;
/// Pitch change (as |ln ratio|) between consecutive frames that counts as
/// a jump: 0.1 is about 1.7 semitones within one hop, far faster than a
/// voice glides or a vibrato swings.
const PITCH_JUMP: f32 = 0.1;
/// Share of consecutive frame pairs with a [`PITCH_JUMP`] above which the
/// "pitch track" is taken to be random (cricket pulses: ~0.3, tones: 0).
const BROADBAND_MIN_JUMP_SHARE: f32 = 0.15;

/// Whether the marked sound is a band of noise rather than a tone. Both
/// signs must agree, so tonal sounds are never misjudged on one alone:
/// - the tracked peak carries only a small share of the band's energy
///   above the noise (`weighted` is the presence-weighted magnitude), and
/// - the peak jumps around between consecutive frames.
///
/// A rich voice marked over many overtones fails the first test but not
/// the second; a fast chirp may jump, but then it holds its band's energy.
fn is_broadband(
    pitch_track: &[Option<f32>],
    active: &[usize],
    weighted: &[Vec<f32>],
    (k0, k1): (usize, usize),
    tolerance: f32,
) -> bool {
    // Mean share of the energy near the peak over the active frames.
    let mut share_sum = 0.0f32;
    let mut frames = 0usize;
    for &t in active {
        let Some(p) = pitch_track[t] else { continue };
        let row = &weighted[t];
        let total: f32 = row[k0..=k1].iter().map(|v| v * v).sum();
        if total <= 0.0 {
            continue;
        }
        let (lo, hi) = band(p, (p * tolerance).max(2.0), row.len());
        let near: f32 = row[lo.max(k0)..=hi.min(k1)].iter().map(|v| v * v).sum();
        share_sum += near / total;
        frames += 1;
    }
    if frames < 4 {
        return false; // too little to judge; keep the tonal analysis
    }
    let peak_share = share_sum / frames as f32;
    // Share of neighbouring active frames whose peaks jump.
    let (mut pairs, mut jumps) = (0usize, 0usize);
    for w in active.windows(2) {
        if w[1] != w[0] + 1 {
            continue;
        }
        if let (Some(a), Some(b)) = (pitch_track[w[0]], pitch_track[w[1]]) {
            pairs += 1;
            if (b.max(0.5) / a.max(0.5)).ln().abs() > PITCH_JUMP {
                jumps += 1;
            }
        }
    }
    let jump_share = jumps as f32 / pairs.max(1) as f32;
    peak_share < BROADBAND_MAX_PEAK_SHARE && jump_share > BROADBAND_MIN_JUMP_SHARE
}

/// Extent of a noise band: the bins in `k0 ..= k1` between which lie 95 %
/// of the band's energy above the noise over the active frames (2.5 % cut
/// off at either end, so a weak neighbour overlapping the selection's edge
/// does not stretch the band). `None` if there is no such energy.
fn noise_band(weighted: &[Vec<f32>], active: &[usize], (k0, k1): (usize, usize)) -> Option<(usize, usize)> {
    let energy: Vec<f32> = (k0..=k1)
        .map(|k| active.iter().map(|&t| weighted[t][k] * weighted[t][k]).sum())
        .collect();
    let total: f32 = energy.iter().sum();
    if total <= 0.0 {
        return None;
    }
    // First bin at which the cumulative energy reaches `q` of the total.
    let quantile = |q: f32| {
        let mut acc = 0.0f32;
        energy.iter().position(|&e| {
            acc += e;
            acc >= q * total
        })
    };
    Some((k0 + quantile(0.025)?, k0 + quantile(0.975)?))
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
            &[],
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

    /// Stereo "duet": a bird gliding from 300 to 450 Hz with its octave,
    /// placed left (right channel -8 dB, 3 samples later), and a second
    /// source that glides exactly along the bird's 3rd harmonic but is
    /// placed right (+8 dB, no delay). Returns the channels and the mono mix.
    fn stereo_duet(sr: u32, secs: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
        let n = (sr as f32 * secs) as usize;
        let (mut bird, mut other) = (vec![0.0f32; n], vec![0.0f32; n]);
        let mut phase = 0.0f32;
        for i in 0..n {
            let t = i as f32 / sr as f32;
            phase += 2.0 * std::f32::consts::PI * (300.0 + 150.0 * t / secs) / sr as f32;
            let env = (std::f32::consts::PI * t / secs).sin();
            bird[i] = env * (0.5 * phase.sin() + 0.25 * (2.0 * phase).sin());
            other[i] = env * 0.15 * (3.0 * phase + 0.7).sin();
        }
        let place = |src: &[f32], gain_db: f32, delay: usize, ch: &mut [Vec<f32>]| {
            let g = 10f32.powf(gain_db / 20.0);
            for i in 0..n {
                ch[0][i] += src[i];
                if i + delay < n {
                    ch[1][i + delay] += g * src[i];
                }
            }
        };
        let mut ch = vec![vec![0.0f32; n]; 2];
        place(&bird, -8.0, 3, &mut ch);
        place(&other, 8.0, 0, &mut ch);
        let mut rng = 99u64;
        for c in ch.iter_mut() {
            for v in c.iter_mut() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                *v += ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.002;
            }
        }
        let mono = ch[0].iter().zip(&ch[1]).map(|(a, b)| 0.5 * (a + b)).collect();
        (ch, mono)
    }

    #[test]
    fn stereo_partial_from_another_direction_is_not_attributed() {
        let sr = 16000;
        let (ch, mono) = stereo_duet(sr, 2.0);
        let cfg = StftConfig {
            fft_size: 1024,
            hop: 256,
        };
        let stft = Stft::new(cfg);
        let sel = Selection {
            start_s: 0.3,
            end_s: 1.7,
            low_hz: 280.0,
            high_hz: 470.0,
        };
        let run = |channels: &[Vec<f32>]| {
            analyze_selection(
                &mono,
                channels,
                sr,
                &stft,
                &sel,
                &NoiseEstimate::default(),
                &AnalysisParams::default(),
            )
        };
        // Weight of the harmonic mask at m·f0 in the middle of the selection.
        let at = |a: &RegionAnalysis, m: f32| {
            let mid = (a.frames / 2) as i64 + a.first_frame;
            let f0 = 300.0 + 150.0 * (mid as f32 * cfg.hop as f32 / sr as f32) / 2.0;
            a.frame(mid).unwrap().1[cfg.hz_to_bin(m * f0, sr).round() as usize]
        };
        // From the mix alone the other source looks like the 3rd harmonic.
        let blind = run(&[]);
        assert!(at(&blind, 2.0) > 0.5 && at(&blind, 3.0) > 0.5, "mono analysis changed");
        // Its level ratio and delay give it away.
        let seeing = run(&ch);
        assert!(at(&seeing, 2.0) > 0.5, "octave lost: {}", at(&seeing, 2.0));
        assert!(at(&seeing, 3.0) < 0.1, "other source kept: {}", at(&seeing, 3.0));
        let sig = seeing
            .template
            .as_ref()
            .and_then(|t| t.spatial.clone())
            .expect("direction");
        assert!((sig.pairs[0].level_db + 8.0).abs() < 1.0, "{sig:?}");
        assert!((sig.pairs[0].delay_samples - 3.0).abs() <= 0.5, "{sig:?}");
    }

    /// `src` recorded by two spaced microphones in a room: the second one
    /// hears it `delay` samples later and `gain_db` louder, and each one
    /// picks up its own reflections (an exponentially decaying random tail,
    /// `reverb` times as strong as the direct sound, 60 ms time constant;
    /// `seed` picks the room). That is what real recordings look like: per
    /// bin, the level difference and the phase between the channels scatter
    /// around the direct sound's values.
    fn in_a_room(src: &[f32], sr: u32, gain_db: f32, delay: usize, reverb: f32, seed: u64) -> Vec<Vec<f32>> {
        let n = src.len();
        let tail = (0.25 * sr as f32) as usize;
        let mut rng = seed;
        (0..2)
            .map(|c| {
                let mut ir = vec![0.0f32; tail];
                ir[c * delay] = if c == 0 { 1.0 } else { 10f32.powf(gain_db / 20.0) };
                for (i, v) in ir.iter_mut().enumerate().skip(c * delay + 1) {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    // uniform in -0.5..0.5, scaled to unit variance by 3.4
                    let r = (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    *v += reverb * 3.4 * r * (-(i as f32) / (0.06 * sr as f32)).exp();
                }
                let mut y = vec![0.0f32; n];
                for (i, &x) in src.iter().enumerate().filter(|(_, x)| **x != 0.0) {
                    for (j, &h) in ir.iter().enumerate().take(n - i) {
                        y[i + j] += x * h;
                    }
                }
                y
            })
            .collect()
    }

    #[test]
    fn direction_check_works_in_a_reverberant_room() {
        // Regression test: in real (stereo, reverberant) recordings the
        // direction check used to veto the harmonics of the marked sound
        // itself, so nothing turned red in the spectrogram, while mono files
        // worked. A neighbour from elsewhere must still be told apart.
        let sr = 16000;
        // The sources of `stereo_duet`, before they are placed.
        let n = 2 * sr as usize;
        let (mut bird, mut other) = (vec![0.0f32; n], vec![0.0f32; n]);
        let mut phase = 0.0f32;
        for i in 0..n {
            let t = i as f32 / sr as f32;
            phase += 2.0 * std::f32::consts::PI * (300.0 + 150.0 * t / 2.0) / sr as f32;
            let env = (std::f32::consts::PI * t / 2.0).sin();
            bird[i] = env * (0.5 * phase.sin() + 0.25 * (2.0 * phase).sin());
            other[i] = env * 0.15 * (3.0 * phase + 0.7).sin();
        }
        let b = in_a_room(&bird, sr, -2.0, 5, 0.03, 7);
        let o = in_a_room(&other, sr, 14.0, 0, 0.03, 8);
        let ch: Vec<Vec<f32>> = b
            .iter()
            .zip(&o)
            .map(|(x, y)| x.iter().zip(y).map(|(a, b)| a + b).collect())
            .collect();
        let mono: Vec<f32> = ch[0].iter().zip(&ch[1]).map(|(a, b)| 0.5 * (a + b)).collect();
        let cfg = StftConfig {
            fft_size: 1024,
            hop: 256,
        };
        let stft = Stft::new(cfg);
        let sel = Selection {
            start_s: 0.3,
            end_s: 1.7,
            low_hz: 280.0,
            high_hz: 470.0,
        };
        let a = analyze_selection(
            &mono,
            &ch,
            sr,
            &stft,
            &sel,
            &NoiseEstimate::default(),
            &AnalysisParams::default(),
        );
        // Mean harmonic mask along the m-th partial of the pitch track.
        let along = |m: f32| {
            let (mut sum, mut count) = (0.0, 0);
            for t in 0..a.frames {
                let Some(p) = a.pitch_track[t] else { continue };
                let k = (m * p).round() as usize;
                sum += a.harmonics[t][k - 1..=k + 1].iter().cloned().fold(0.0f32, f32::max);
                count += 1;
            }
            sum / count.max(1) as f32
        };
        assert!(along(2.0) > 0.7, "own octave vetoed: mean mask {:.2}", along(2.0));
        assert!(along(3.0) < 0.15, "neighbour kept: mean mask {:.2}", along(3.0));
    }

    /// Issue #16: a steady tone (constant envelope inside the selection)
    /// with a clean octave at high SNR. The octave belongs to the sound as
    /// surely as the fundamental, so its harmonic mask must be close to 1,
    /// not merely above the threshold.
    #[test]
    fn steady_tone_octave_gets_a_full_mask() {
        let sr = 16000;
        let n = 3 * sr as usize;
        let mut rng = 777u64;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.04;
                let tone = if (1.0..2.0).contains(&t) {
                    let w = 2.0 * std::f32::consts::PI * 660.0 * t;
                    0.4 * w.sin() + 0.2 * (2.0 * w).sin()
                } else {
                    0.0
                };
                tone + noise
            })
            .collect();
        let cfg = StftConfig {
            fft_size: 1024,
            hop: 256,
        };
        let stft = Stft::new(cfg);
        // Same selection as the CLI test of #16: only the fundamental, and
        // well inside the tone, so the envelope is flat.
        let sel = Selection {
            start_s: 1.1,
            end_s: 1.9,
            low_hz: 550.0,
            high_hz: 780.0,
        };
        let a = analyze_selection(
            &x,
            &[],
            sr,
            &stft,
            &sel,
            &NoiseEstimate::default(),
            &AnalysisParams::default(),
        );
        assert!(
            a.partials.iter().any(|p| p.ratio == 2.0),
            "octave not found: {:?}",
            a.partials
        );
        let k_oct = (1320.0 * cfg.fft_size as f32 / sr as f32).round() as usize;
        let mid = a.first_frame + (a.frames / 2) as i64;
        let (_, harm, _) = a.frame(mid).unwrap();
        let m = harm[k_oct - 1..=k_oct + 1].iter().cloned().fold(0.0f32, f32::max);
        assert!(m > 0.9, "octave mask only {m}");
    }

    /// Pseudo-random numbers in -0.5..0.5 (a fixed LCG, so tests repeat).
    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }

    /// Insect-like pulses (issue #24): band-limited noise from 6 to 20 kHz,
    /// switched on and off at 6 Hz, over faint white noise, 48 kHz.
    fn noise_pulses(secs: f32) -> Vec<f32> {
        let sr = 48000.0;
        let n = (sr * secs) as usize;
        let mut rng = 99u64;
        // Band-limited noise as a sum of many sines with random frequencies
        // and phases inside the band.
        let partials: Vec<(f32, f32)> = (0..300)
            .map(|_| (13000.0 + 14000.0 * lcg(&mut rng), std::f32::consts::TAU * lcg(&mut rng)))
            .collect();
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let on = (t * 6.0).fract() < 0.5;
                let band = if on {
                    partials
                        .iter()
                        .map(|&(f, ph)| (std::f32::consts::TAU * f * t + ph).sin())
                        .sum::<f32>()
                        * 0.01
                } else {
                    0.0
                };
                band + 0.002 * lcg(&mut rng)
            })
            .collect()
    }

    #[test]
    fn noise_pulses_are_broadband_without_pitch_or_partials() {
        let x = noise_pulses(2.0);
        let stft = Stft::new(StftConfig::default());
        let sel = Selection {
            start_s: 0.2,
            end_s: 1.8,
            low_hz: 2000.0,
            high_hz: 22000.0,
        };
        let a = analyze_selection(
            &x,
            &[],
            48000,
            &stft,
            &sel,
            &NoiseEstimate::default(),
            &AnalysisParams::default(),
        );
        let (lo, hi) = a.broadband.expect("noise band taken for a tone");
        let hz = |k: usize| StftConfig::default().bin_to_hz(k as f32, 48000);
        assert!((5000.0..8000.0).contains(&hz(lo)), "band starts at {} Hz", hz(lo));
        assert!((18500.0..21000.0).contains(&hz(hi)), "band ends at {} Hz", hz(hi));
        assert!(a.partials.is_empty(), "partials in noise: {:?}", a.partials);
        assert!(a.pitch_track.iter().all(Option::is_none));
        assert!(a.harmonics.iter().flatten().all(|&m| m == 0.0));
        // The band itself is still the marked sound.
        let k = (10000.0 * 2048.0 / 48000.0) as usize;
        assert!(a.selected.iter().any(|f| f[k] > 0.9));
        assert!(a.model.is_none());
    }

    #[test]
    fn harmonic_tone_with_vibrato_stays_tonal() {
        let sr = 48000;
        let n = 2 * sr as usize;
        let mut rng = 5u64;
        let mut phase = 0.0f32;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                // 5.5 Hz vibrato of ±3 % around 800 Hz, three overtones.
                let f0 = 800.0 * (1.0 + 0.03 * (std::f32::consts::TAU * 5.5 * t).sin());
                phase += std::f32::consts::TAU * f0 / sr as f32;
                0.4 * phase.sin() + 0.2 * (2.0 * phase).sin() + 0.1 * (3.0 * phase).sin() + 0.002 * lcg(&mut rng)
            })
            .collect();
        let stft = Stft::new(StftConfig::default());
        // A wide selection over the fundamental and its overtones.
        let sel = Selection {
            start_s: 0.2,
            end_s: 1.8,
            low_hz: 600.0,
            high_hz: 2800.0,
        };
        let a = analyze_selection(
            &x,
            &[],
            sr,
            &stft,
            &sel,
            &NoiseEstimate::default(),
            &AnalysisParams::default(),
        );
        assert!(a.broadband.is_none(), "tone taken for noise");
        let ratios: Vec<f32> = a.partials.iter().map(|p| p.ratio).collect();
        assert!(ratios.contains(&2.0) && ratios.contains(&3.0), "partials {ratios:?}");
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
            &[],
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
            &[],
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
