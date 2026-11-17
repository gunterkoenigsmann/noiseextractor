//! Finding other occurrences of a marked sound in the whole recording.
//!
//! Birds repeat their calls, machines repeat their clicks, a dog barks more
//! than once. Once the user has marked one instance and the analysis has
//! worked out which time-frequency cells belong to it, that pattern is used
//! as a *template*: a small spectro-temporal patch of log levels. The
//! template is slid along the recording, and wherever the recording's patch
//! correlates strongly with it (normalised cross-correlation, allowing a
//! small shift in frequency), another occurrence is assumed.
//!
//! Why a 2-D template instead of a single spectrum (as in plain NMF)? A
//! chirp whose pitch glides is a *shape* in the spectrogram; its average
//! spectrum is a smear that matches lots of unrelated sounds. Matching the
//! shape keeps the drifting pitch, the overtones and their timing together.
//!
//! The mask of an occurrence is the template's membership pattern moved to
//! the new place, weighted by how far the energy there is above the noise,
//! so quiet parts of a found occurrence are treated as gently as in the
//! original.
//!
//! ## Chunk independence
//!
//! The engine processes the recording in chunks, possibly in any order. The
//! detection therefore must give the same answer for an offset no matter
//! which chunk asks. That holds because the features only depend on the
//! global STFT grid and on the noise spectrum stored with the template (not
//! on a per-chunk estimate), and because peak picking (non-maximum
//! suppression) looks at a fixed neighbourhood that [`find_in_range`] always
//! computes completely.

use crate::rhythm::Rhythm;
use crate::spatial::{agreement, measure, Cells, Signature};
use crate::stft::Stft;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Tuning of the search for similar occurrences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SimilarParams {
    /// Correlation (0..1) above which a place counts as another occurrence.
    pub threshold: f32,
    /// Largest frequency shift in bins tried when matching; small values
    /// tolerate slightly higher or lower repetitions.
    pub max_shift_bins: usize,
    /// How much the spatial cues of multi-channel recordings count (0..1):
    /// the shape score is multiplied by `1 - w + w·agreement`, so with
    /// w = 1 a repetition from a different place is rejected completely,
    /// with w = 0 the direction is ignored. No effect on mono files.
    pub spatial_weight: f32,
    /// Level difference (dB) at which the spatial agreement drops to e^-½.
    pub level_tolerance_db: f32,
    /// Delay difference (samples) at which the agreement drops to e^-½.
    pub delay_tolerance_samples: f32,
    /// Pulsed sounds only (see [`crate::rhythm`]): how far (dB) the band
    /// must be above the noise floor on average for a repetition.
    pub min_level_db: f32,
}

impl Default for SimilarParams {
    /// Threshold 0.6 and ±2 bins (≈ ±47 Hz at 48 kHz with 2048-point FFTs).
    fn default() -> Self {
        Self {
            threshold: 0.6,
            max_shift_bins: 2,
            spatial_weight: 0.6,
            level_tolerance_db: 3.0,
            delay_tolerance_samples: 2.0,
            min_level_db: 3.0,
        }
    }
}

/// Cells of the analysis whose mask exceeds this value form the template's
/// time and frequency extent.
const MEMBER_LEVEL: f32 = 0.3;
/// Extra bins around the member cells, so the template also "knows" where
/// the sound has no energy.
const BIN_MARGIN: usize = 3;

/// The spectro-temporal pattern of a marked sound.
#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    /// Global frame of the template's first frame (inside the analysis).
    pub first_frame: i64,
    /// Number of frames.
    pub frames: usize,
    /// Lowest bin of the patch.
    pub k0: usize,
    /// Highest bin of the patch (inclusive).
    pub k1: usize,
    /// Mean-free feature values, frame-major (`frame * width + bin - k0`).
    values: Vec<f32>,
    /// Euclidean norm of `values`.
    norm: f32,
    /// Noise power per bin (all bins) used for the features, so every chunk
    /// computes identical features.
    pub noise: Vec<f32>,
    /// Membership of each patch cell (union of all parts), frame-major like
    /// `values`; selects the cells used for spatial measurements.
    members: Vec<f32>,
    /// Where the marked sound came from (multi-channel recordings only;
    /// set by [`Template::attach_spatial`]).
    pub spatial: Option<Signature>,
    /// Largest inter-channel delay searched when measuring directions, in
    /// samples (the same for the marked sound and its repetitions).
    pub max_delay_samples: f32,
    /// The rhythm of a pulsed sound. When set, repetitions are found by
    /// rhythm and band shape (see [`crate::rhythm`]) instead of the 2-D
    /// correlation, and may be longer or shorter than the template.
    pub rhythm: Option<Rhythm>,
    /// Largest membership of each patch bin over all template frames: the
    /// band a pulsed sound occupies, used for the direction of its
    /// repetitions.
    band: Vec<f32>,
}

/// One place where the template matched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Occurrence {
    /// Global frame where the matched patch starts.
    pub frame: i64,
    /// Length in frames: the template's length, or for pulsed sounds the
    /// length of the found burst.
    pub frames: usize,
    /// Frequency shift in bins relative to the template.
    pub shift: i32,
    /// Combined score (0..1): shape correlation, reduced by spatial
    /// disagreement in multi-channel recordings.
    pub score: f32,
    /// Agreement of the occurrence's direction with the marked sound's
    /// (1 = same place), `None` when no spatial cues were available.
    pub spatial_agreement: Option<f32>,
}

/// Feature of one bin: log power with the noise floor added, so silence
/// and noise map to a flat floor and only real structure varies.
fn feature(power: f32, noise: f32) -> f32 {
    (power + noise + 1e-20).ln()
}

impl Template {
    /// Width of the patch in bins.
    fn width(&self) -> usize {
        self.k1 - self.k0 + 1
    }

    /// Builds the template from the analysed power frames `power`
    /// (starting at global frame `first_frame`), the noise spectrum and the
    /// union of all membership masks of the analysis. Returns `None` when
    /// the masks contain nothing (nothing to look for).
    pub fn build(power: &[Vec<f32>], first_frame: i64, noise: &[f32], members: &[Vec<f32>]) -> Option<Self> {
        let bins = noise.len();
        let active: Vec<usize> = (0..members.len())
            .filter(|&t| members[t].iter().any(|&m| m > MEMBER_LEVEL))
            .collect();
        let (&t0, &t1) = (active.first()?, active.last()?);
        let mut lo = bins;
        let mut hi = 0;
        for row in &members[t0..=t1] {
            for (k, &m) in row.iter().enumerate() {
                if m > MEMBER_LEVEL {
                    lo = lo.min(k);
                    hi = hi.max(k);
                }
            }
        }
        let k0 = lo.saturating_sub(BIN_MARGIN);
        let k1 = (hi + BIN_MARGIN).min(bins - 1);
        let mut values = Vec::with_capacity((t1 - t0 + 1) * (k1 - k0 + 1));
        for frame in &power[t0..=t1] {
            values.extend((k0..=k1).map(|k| feature(frame[k], noise[k])));
        }
        let cell_members: Vec<f32> = members[t0..=t1]
            .iter()
            .flat_map(|row| row[k0..=k1].iter().copied())
            .collect();
        let width = k1 - k0 + 1;
        let band: Vec<f32> = (0..width)
            .map(|i| cell_members.chunks(width).map(|row| row[i]).fold(0.0, f32::max))
            .collect();
        let rhythm = Rhythm::build(&power[t0..=t1], noise, k0, k1);
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        values.iter_mut().for_each(|v| *v -= mean);
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Flat patch (RMS deviation below 0.001 log units, i.e. rounding
        // noise): nothing distinctive to match.
        if norm <= 1e-3 * (values.len() as f32).sqrt() {
            return None;
        }
        Some(Self {
            first_frame: first_frame + t0 as i64,
            frames: t1 - t0 + 1,
            k0,
            k1,
            values,
            norm,
            noise: noise.to_vec(),
            members: cell_members,
            spatial: None,
            max_delay_samples: 0.0,
            rhythm,
            band,
        })
    }

    /// The template's member cells placed at global frame `frame`, shifted
    /// by `shift` bins, for spatial measurements.
    fn cells(&self, frame: i64, shift: i32) -> Option<Cells<'_>> {
        let k0 = self.k0 as i64 + shift as i64;
        if k0 < 0 || k0 as usize + self.width() > self.noise.len() {
            return None;
        }
        Some(Cells {
            first_frame: frame,
            frames: self.frames,
            k0: k0 as usize,
            weights: &self.members,
        })
    }

    /// Measures where the marked sound came from, so repetitions can be
    /// checked for the same direction, searching delays up to
    /// ±`max_delay_samples`. Leaves `spatial` at `None` for mono recordings.
    pub fn attach_spatial(&mut self, channels: &[Vec<f32>], stft: &Stft, max_delay_samples: f32) {
        self.max_delay_samples = max_delay_samples;
        self.spatial = self
            .cells(self.first_frame, 0)
            .and_then(|cells| measure(channels, stft, &cells, max_delay_samples));
    }

    /// Normalised cross-correlation of the template with the patch of
    /// `features` starting at row `row` and shifted by `shift` bins.
    /// Returns `None` when the shifted patch leaves the spectrum.
    fn correlate(&self, features: &[Vec<f32>], row: usize, shift: i32) -> Option<f32> {
        let w = self.width();
        let k0 = self.k0 as i64 + shift as i64;
        if k0 < 0 || (k0 as usize + w) > self.noise.len() {
            return None;
        }
        let k0 = k0 as usize;
        // Σz·c, Σc and Σc²; because Σz = 0 the covariance is just Σz·c.
        let (mut dot, mut sum, mut sum_sq) = (0.0f64, 0.0f64, 0.0f64);
        for t in 0..self.frames {
            let f = &features[row + t][k0..k0 + w];
            let z = &self.values[t * w..(t + 1) * w];
            for (&c, &z) in f.iter().zip(z) {
                dot += (z * c) as f64;
                sum += c as f64;
                sum_sq += (c * c) as f64;
            }
        }
        let n = (self.frames * w) as f64;
        let var = sum_sq - sum * sum / n;
        if var <= 1e-9 {
            return Some(0.0);
        }
        Some((dot / (self.norm as f64 * var.sqrt())) as f32)
    }
}

/// Finds all occurrences that overlap the global frame range `range`.
///
/// Runs its own STFT on `mono` over the frames it needs, so it can be
/// called from any chunk. An occurrence at offset `o` is reported if its
/// best score (over the allowed shifts) reaches the threshold and is the
/// maximum within ±half a template length; ties go to the earlier offset.
///
/// With several `channels` and a template that knows its direction, every
/// offset whose shape score reaches the threshold is also checked for
/// coming from the same place (see [`crate::spatial`]); the combined score
/// is used for the threshold and the peak picking. This is decided per
/// offset, so chunks still agree with each other.
pub fn find_in_range(
    template: &Template,
    mono: &[f32],
    channels: &[Vec<f32>],
    stft: &Stft,
    range: Range<i64>,
    params: &SimilarParams,
) -> Vec<Occurrence> {
    if let Some(rhythm) = &template.rhythm {
        return find_rhythmic(template, rhythm, mono, channels, stft, range, params);
    }
    let t = template.frames as i64;
    let radius = (t / 2).max(1);
    let total_frames = stft.config().frame_count(mono.len()) as i64;
    // Offsets whose occurrence overlaps `range`, then those needed for
    // non-maximum suppression around them.
    let cand = (range.start - t + 1).max(0)..range.end.min(total_frames - t + 1);
    if cand.is_empty() {
        return Vec::new();
    }
    let scored = (cand.start - radius).max(0)..(cand.end + radius).min(total_frames - t + 1);
    let feat_first = scored.start;
    let feat_len = (scored.end - scored.start + t - 1).max(0) as usize;
    let power = stft.analyze(mono, feat_first, feat_len).power();
    let features: Vec<Vec<f32>> = power
        .iter()
        .map(|f| f.iter().zip(&template.noise).map(|(&p, &n)| feature(p, n)).collect())
        .collect();

    let shifts = -(params.max_shift_bins as i32)..=(params.max_shift_bins as i32);
    let shape_scores: Vec<(f32, i32)> = (scored.start..scored.end)
        .map(|o| {
            let row = (o - feat_first) as usize;
            shifts
                .clone()
                .filter_map(|s| template.correlate(&features, row, s).map(|c| (c, s)))
                // Prefer the smallest shift on ties so results are stable.
                .fold((f32::MIN, 0i32), |best, (c, s)| {
                    if c > best.0 || (c == best.0 && s.abs() < best.1.abs()) {
                        (c, s)
                    } else {
                        best
                    }
                })
        })
        .collect();
    let spatial = template
        .spatial
        .as_ref()
        .filter(|_| channels.len() >= 2 && params.spatial_weight > 0.0);
    let scores: Vec<(f32, i32, Option<f32>)> = shape_scores
        .into_iter()
        .enumerate()
        .map(|(i, (score, shift))| {
            let Some(reference) = spatial.filter(|_| score >= params.threshold) else {
                return (score, shift, None);
            };
            let o = scored.start + i as i64;
            let found = template
                .cells(o, shift)
                .and_then(|cells| measure(channels, stft, &cells, template.max_delay_samples));
            let agree = found.map_or(0.0, |sig| {
                agreement(
                    reference,
                    &sig,
                    params.level_tolerance_db,
                    params.delay_tolerance_samples,
                )
            });
            let w = params.spatial_weight.clamp(0.0, 1.0);
            (score * (1.0 - w + w * agree), shift, Some(agree))
        })
        .collect();

    let at = |o: i64| scores[(o - scored.start) as usize];
    let mut out = Vec::new();
    for o in cand {
        let (score, shift, spatial_agreement) = at(o);
        if score < params.threshold {
            continue;
        }
        let lo = (o - radius).max(scored.start);
        let hi = (o + radius).min(scored.end - 1);
        // Strictly greater than earlier neighbours, at least as great as
        // later ones: exactly one offset of a plateau wins.
        let is_peak = (lo..o).all(|p| at(p).0 < score) && (o + 1..=hi).all(|p| at(p).0 <= score);
        if is_peak {
            out.push(Occurrence {
                frame: o,
                frames: template.frames,
                shift,
                score,
                spatial_agreement,
            });
        }
    }
    out
}

/// Rhythm scores of the window starts `lo .. hi` (see [`Rhythm::score`]).
fn rhythm_scores(
    template: &Template,
    rhythm: &Rhythm,
    mono: &[f32],
    stft: &Stft,
    lo: i64,
    hi: i64,
    params: &SimilarParams,
) -> Vec<f32> {
    let w = rhythm.window;
    let power = stft.analyze(mono, lo, (hi - lo) as usize + w - 1).power();
    let levels = rhythm.levels(&power, &template.noise);
    (0..(hi - lo) as usize)
        .map(|i| rhythm.score(&levels, i, params.min_level_db))
        .collect()
}

/// [`find_in_range`] for pulsed sounds.
///
/// Every window start whose rhythm score reaches the threshold is a hit;
/// hits less than a window apart belong to the same burst, which becomes
/// one occurrence from its first hit to the end of its last window. A
/// burst may reach far beyond `range`, so the scanned span grows until
/// both of its ends are at least a window away from any hit (or at the
/// recording's ends). Every chunk therefore sees the same bursts.
fn find_rhythmic(
    template: &Template,
    rhythm: &Rhythm,
    mono: &[f32],
    channels: &[Vec<f32>],
    stft: &Stft,
    range: Range<i64>,
    params: &SimilarParams,
) -> Vec<Occurrence> {
    let w = rhythm.window as i64;
    let gap = w;
    let total_frames = stft.config().frame_count(mono.len()) as i64;
    // Window starts run over 0 ..= last.
    let last = total_frames - w;
    if last < 0 {
        return Vec::new();
    }
    let mut lo = (range.start - w - gap).clamp(0, last);
    let mut hi = (range.end + gap).clamp(lo + 1, last + 1);
    let mut grow = 4 * w;
    let scores = loop {
        let scores = rhythm_scores(template, rhythm, mono, stft, lo, hi, params);
        let hit = |i: usize| scores[i] >= params.threshold;
        let edge = (gap as usize).min(scores.len());
        let open_left = lo > 0 && (0..edge).any(hit);
        let open_right = hi <= last && (scores.len() - edge..scores.len()).any(hit);
        if !open_left && !open_right {
            break scores;
        }
        if open_left {
            lo = (lo - grow).max(0);
        }
        if open_right {
            hi = (hi + grow).min(last + 1);
        }
        grow *= 2;
    };
    // Group hits into bursts: (first start, last start, best score).
    let mut bursts: Vec<(i64, i64, f32)> = Vec::new();
    for (i, &s) in scores.iter().enumerate() {
        if s < params.threshold {
            continue;
        }
        let o = lo + i as i64;
        match bursts.last_mut() {
            Some(b) if o - b.1 <= gap => {
                b.1 = o;
                b.2 = b.2.max(s);
            }
            _ => bursts.push((o, o, s)),
        }
    }
    let spatial = template
        .spatial
        .as_ref()
        .filter(|_| channels.len() >= 2 && params.spatial_weight > 0.0);
    bursts
        .into_iter()
        .map(|(a, b, score)| (a, (b - a + w) as usize, score))
        .filter(|&(a, frames, _)| a < range.end && a + frames as i64 > range.start)
        .filter_map(|(a, frames, score)| {
            let mut occ = Occurrence {
                frame: a,
                frames,
                shift: 0,
                score,
                spatial_agreement: None,
            };
            if let Some(reference) = spatial {
                // The direction of the burst's band, measured over at most
                // a template length (enough cells, bounded cost).
                let n = frames.min(template.frames.max(rhythm.window));
                let weights: Vec<f32> = (0..n).flat_map(|_| template.band.iter().copied()).collect();
                let cells = Cells {
                    first_frame: a,
                    frames: n,
                    k0: template.k0,
                    weights: &weights,
                };
                let agree = measure(channels, stft, &cells, template.max_delay_samples).map_or(0.0, |sig| {
                    agreement(
                        reference,
                        &sig,
                        params.level_tolerance_db,
                        params.delay_tolerance_samples,
                    )
                });
                let wgt = params.spatial_weight.clamp(0.0, 1.0);
                occ.score = score * (1.0 - wgt + wgt * agree);
                occ.spatial_agreement = Some(agree);
            }
            (occ.score >= params.threshold).then_some(occ)
        })
        .collect()
}

impl Occurrence {
    /// First frame after the occurrence.
    pub fn end(&self) -> i64 {
        self.frame + self.frames as i64
    }

    /// Whether this is the marked sound itself: it matches the template's
    /// own position give or take a few frames, or (pulsed sounds) is the
    /// burst that overlaps the template by at least half its length.
    pub fn is_original(&self, template: &Template) -> bool {
        if template.rhythm.is_none() {
            // Same length as the template: give or take a few frames.
            return (self.frame - template.first_frame).abs() <= (template.frames as i64 / 2).max(1);
        }
        let t_end = template.first_frame + template.frames as i64;
        let overlap = self.end().min(t_end) - self.frame.max(template.first_frame);
        2 * overlap >= template.frames as i64
    }

    /// Whether the occurrence adds anything at global frame `frame`: the
    /// frame lies inside it, and not where the marked sound itself already
    /// covers it. A burst of a pulsed sound that contains the marked part
    /// still extends it to the burst's other frames.
    pub fn applies_to(&self, template: &Template, frame: i64) -> bool {
        if frame < self.frame || frame >= self.end() {
            return false;
        }
        if template.rhythm.is_some() {
            !(template.first_frame..template.first_frame + template.frames as i64).contains(&frame)
        } else {
            !self.is_original(template)
        }
    }

    /// The template frame (0-based) that corresponds to global frame
    /// `frame` of this occurrence. Bursts of pulsed sounds may be longer
    /// than the template; they wrap around.
    pub fn template_frame(&self, template: &Template, frame: i64) -> usize {
        ((frame - self.frame).max(0) as usize) % template.frames.max(1)
    }
}

/// Soft weight of an occurrence: 0 at the threshold, 1 at a perfect match.
pub fn occurrence_weight(score: f32, params: &SimilarParams) -> f32 {
    crate::stats::smoothstep(
        params.threshold,
        params.threshold + (1.0 - params.threshold) * 0.5,
        score,
    )
}

/// Adds the mask of `occ` for global frame `frame` to `out` (maximum).
///
/// `member` gives the membership of the template's cells (the union of the
/// enabled parts of the original analysis) for a template-relative frame;
/// `power` is the power of `frame` at the target position, used to weight
/// the mask by how far the energy there is above the template's noise.
pub fn add_occurrence_mask(
    template: &Template,
    occ: &Occurrence,
    params: &SimilarParams,
    frame: i64,
    power: &[f32],
    member: &dyn Fn(usize, usize) -> f32,
    out: &mut [f32],
) {
    let rel = frame - occ.frame;
    if rel < 0 || rel >= occ.frames as i64 {
        return;
    }
    let weight = occurrence_weight(occ.score, params);
    // Pulsed sounds: the pulses of a repetition do not line up with the
    // template's, so every frame gets the band the sound occupies (its
    // membership over the whole template); the presence term below keeps
    // the gaps between pulses free. Up to 32 template frames are sampled.
    let stride = (template.frames / 32).max(1);
    let membership_of = |k: usize| -> f32 {
        if template.rhythm.is_some() {
            (0..template.frames)
                .step_by(stride)
                .map(|t| member(t, k))
                .fold(0.0, f32::max)
        } else {
            member(rel as usize, k)
        }
    };
    for k in template.k0..=template.k1 {
        let m = membership_of(k);
        if m <= 0.0 {
            continue;
        }
        let target = k as i64 + occ.shift as i64;
        if target < 0 || target as usize >= out.len() {
            continue;
        }
        let target = target as usize;
        // Cells that clearly belonged to the original count fully.
        let membership = (m / 0.5).min(1.0);
        let presence = (1.0 - 2.0 * template.noise[target] / power[target].max(1e-20)).max(0.0);
        out[target] = out[target].max(weight * membership * presence);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::StftConfig;

    /// A rising chirp from `f0` to `f1` Hz lasting `len` seconds.
    fn chirp(out: &mut [f32], sr: u32, start_s: f32, len: f32, f0: f32, f1: f32, amp: f32) {
        let start = (start_s * sr as f32) as usize;
        let n = (len * sr as f32) as usize;
        let mut phase = 0.0f32;
        for i in 0..n.min(out.len().saturating_sub(start)) {
            let t = i as f32 / n as f32;
            phase += 2.0 * std::f32::consts::PI * (f0 + (f1 - f0) * t) / sr as f32;
            let env = (std::f32::consts::PI * t).sin();
            out[start + i] += amp * env * (phase.sin() + 0.4 * (2.0 * phase).sin());
        }
    }

    /// Noise plus three identical chirps (at 0.5, 2.0 and 3.2 s), a falling
    /// chirp at 1.3 s and a steady tone burst at 2.6 s.
    fn scene(sr: u32) -> Vec<f32> {
        let mut rng = 77u64;
        let mut x: Vec<f32> = (0..4 * sr)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.01
            })
            .collect();
        for s in [0.5, 2.0, 3.2] {
            chirp(&mut x, sr, s, 0.4, 1000.0, 1800.0, 0.3);
        }
        chirp(&mut x, sr, 1.3, 0.4, 1800.0, 1000.0, 0.3);
        chirp(&mut x, sr, 2.6, 0.3, 1400.0, 1400.0, 0.3);
        x
    }

    /// Template from a rectangle around the first chirp, using its power as
    /// membership (good enough to test the matching itself).
    fn template(x: &[f32], stft: &Stft, sr: u32) -> Template {
        let cfg = stft.config();
        let (a, b) = (
            (0.5 * sr as f32 / cfg.hop as f32) as i64,
            (0.9 * sr as f32 / cfg.hop as f32) as i64,
        );
        let power = stft.analyze(x, a, (b - a) as usize).power();
        let noise = vec![1e-3; cfg.bins()];
        let k0 = cfg.hz_to_bin(900.0, sr) as usize;
        let k1 = cfg.hz_to_bin(3700.0, sr) as usize;
        let members: Vec<Vec<f32>> = power
            .iter()
            .map(|f| {
                f.iter()
                    .enumerate()
                    .map(|(k, &p)| if (k0..=k1).contains(&k) && p > 0.5 { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        Template::build(&power, a, &noise, &members).expect("template")
    }

    #[test]
    fn finds_repetitions_but_not_other_shapes() {
        let sr = 16000;
        let x = scene(sr);
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let tpl = template(&x, &stft, sr);
        let all = find_in_range(
            &tpl,
            &x,
            &[],
            &stft,
            0..stft.config().frame_count(x.len()) as i64,
            &SimilarParams::default(),
        );
        let times: Vec<f32> = all.iter().map(|o| o.frame as f32 * 128.0 / sr as f32).collect();
        assert_eq!(all.len(), 3, "expected the three rising chirps, got {times:?}");
        for (t, want) in times.iter().zip([0.5, 2.0, 3.2]) {
            assert!((t - want).abs() < 0.06, "found {t}, expected {want}");
        }
        assert!(all.iter().all(|o| o.score > 0.9));
    }

    #[test]
    fn chunked_search_agrees_with_whole_search() {
        // Any split of the frame range must report the same occurrences,
        // once each (every occurrence overlaps one or two chunks).
        let sr = 16000;
        let x = scene(sr);
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let tpl = template(&x, &stft, sr);
        let total = stft.config().frame_count(x.len()) as i64;
        let p = SimilarParams::default();
        let whole = find_in_range(&tpl, &x, &[], &stft, 0..total, &p);
        let mut pieces = Vec::new();
        let mut a = 0;
        while a < total {
            let b = (a + 37).min(total);
            pieces.extend(find_in_range(&tpl, &x, &[], &stft, a..b, &p));
            a = b;
        }
        pieces.sort_by_key(|o| o.frame);
        pieces.dedup_by_key(|o| o.frame);
        assert_eq!(pieces, whole);
    }

    #[test]
    fn shifted_repetition_is_found_with_its_shift() {
        let sr = 16000;
        let mut x = scene(sr);
        // The same chirp 62.5 Hz (2 bins) higher, before the first one.
        chirp(&mut x, sr, 0.05, 0.4, 1062.5, 1862.5, 0.3);
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let tpl = template(&x, &stft, sr);
        let found = find_in_range(&tpl, &x, &[], &stft, 0..20, &SimilarParams::default());
        assert_eq!(found.len(), 1, "{found:?}");
        // For a gliding chirp a shift in frequency and a shift in time look
        // alike, so the match may trade one for the other; it must however
        // move upwards.
        assert!(found[0].shift > 0, "{found:?}");
    }

    #[test]
    fn the_marked_sound_itself_is_recognised_as_original() {
        let sr = 16000;
        let x = scene(sr);
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let tpl = template(&x, &stft, sr);
        let all = find_in_range(
            &tpl,
            &x,
            &[],
            &stft,
            0..stft.config().frame_count(x.len()) as i64,
            &SimilarParams::default(),
        );
        let originals: Vec<bool> = all.iter().map(|o| o.is_original(&tpl)).collect();
        assert_eq!(originals, vec![true, false, false]);
    }

    #[test]
    fn flat_or_empty_members_give_no_template() {
        let power = vec![vec![1.0; 8]; 4];
        let noise = vec![1.0; 8];
        assert!(Template::build(&power, 0, &noise, &vec![vec![0.0; 8]; 4]).is_none());
        assert!(Template::build(&power, 0, &noise, &vec![vec![1.0; 8]; 4]).is_none());
    }

    #[test]
    fn occurrence_mask_moves_membership_and_respects_presence() {
        let power = vec![vec![1.0, 100.0, 1.0, 1.0]; 2];
        let noise = vec![1.0; 4];
        let mut members = vec![vec![0.0; 4]; 2];
        members[0][1] = 1.0;
        members[1][1] = 1.0;
        let mut tpl = Template::build(&power, 0, &noise, &members).unwrap();
        tpl.k0 = 0;
        tpl.k1 = 3;
        let occ = Occurrence {
            frame: 10,
            frames: tpl.frames,
            shift: 1,
            score: 1.0,
            spatial_agreement: None,
        };
        let mut out = vec![0.0; 4];
        let target_power = [1.0, 1.0, 100.0, 1.0];
        let member = |t: usize, k: usize| members[t][k];
        add_occurrence_mask(
            &tpl,
            &occ,
            &SimilarParams::default(),
            11,
            &target_power,
            &member,
            &mut out,
        );
        assert!(out[2] > 0.9, "{out:?}");
        assert_eq!(out[1], 0.0);
        // Outside the occurrence nothing happens.
        let mut out = vec![0.0; 4];
        add_occurrence_mask(
            &tpl,
            &occ,
            &SimilarParams::default(),
            12,
            &target_power,
            &member,
            &mut out,
        );
        assert_eq!(out, vec![0.0; 4]);
    }

    /// Places `src` into two channels with gain (dB) and delay (whole
    /// samples) on the right channel, adding it to `channels`.
    fn add_panned(channels: &mut [Vec<f32>], src: &[f32], gain_db: f32, delay: usize) {
        let g = 10f32.powf(gain_db / 20.0);
        for i in 0..src.len() {
            channels[0][i] += src[i];
            if i + delay < channels[1].len() {
                channels[1][i + delay] += g * src[i];
            }
        }
    }

    /// Stereo scene: chirps at 0.5 and 2.0 s placed with gain -8 dB / delay
    /// 3 samples on the right channel, a third identical chirp at 3.2 s with
    /// `other_gain_db` / `other_delay`. Returns the channels and the mono mix.
    fn stereo_scene(sr: u32, other_gain_db: f32, other_delay: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let len = 4 * sr as usize;
        let mut left_bird = vec![0.0; len];
        chirp(&mut left_bird, sr, 0.5, 0.4, 1000.0, 1800.0, 0.3);
        chirp(&mut left_bird, sr, 2.0, 0.4, 1000.0, 1800.0, 0.3);
        let mut other_bird = vec![0.0; len];
        chirp(&mut other_bird, sr, 3.2, 0.4, 1000.0, 1800.0, 0.3);
        let mut ch = vec![vec![0.0f32; len]; 2];
        add_panned(&mut ch, &left_bird, -8.0, 3);
        add_panned(&mut ch, &other_bird, other_gain_db, other_delay);
        let mono = ch[0].iter().zip(&ch[1]).map(|(a, b)| 0.5 * (a + b)).collect();
        (ch, mono)
    }

    #[test]
    fn stereo_repetition_with_only_another_delay_is_rejected() {
        // Same level ratio, but the sound arrives 12 instead of 3 samples
        // later on the right: a different place at the same distance ratio.
        let sr = 16000;
        let (ch, mono) = stereo_scene(sr, -8.0, 12);
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let mut tpl = template(&mono, &stft, sr);
        tpl.attach_spatial(&ch, &stft, 16.0);
        let range = 0..stft.config().frame_count(mono.len()) as i64;
        let found = find_in_range(&tpl, &mono, &ch, &stft, range, &SimilarParams::default());
        assert_eq!(found.len(), 2, "{found:?}");
    }

    #[test]
    fn stereo_repetition_from_another_place_is_rejected() {
        let sr = 16000;
        // The marked chirp (0.5 s) and one repetition (2.0 s) come from the
        // left, a third identical chirp (3.2 s) from the right.
        let (ch, mono) = stereo_scene(sr, 8.0, 0);
        let len = mono.len();
        let stft = Stft::new(StftConfig {
            fft_size: 512,
            hop: 128,
        });
        let mut tpl = template(&mono, &stft, sr);
        tpl.attach_spatial(&ch, &stft, 16.0);
        let sig = tpl.spatial.clone().expect("stereo signature");
        assert!((sig.pairs[0].level_db + 8.0).abs() < 1.0, "{sig:?}");
        assert!((sig.pairs[0].delay_samples - 3.0).abs() <= 0.5, "{sig:?}");

        let range = 0..stft.config().frame_count(len) as i64;
        let params = SimilarParams::default();
        let found = find_in_range(&tpl, &mono, &ch, &stft, range.clone(), &params);
        let times: Vec<f32> = found.iter().map(|o| o.frame as f32 * 128.0 / sr as f32).collect();
        assert_eq!(found.len(), 2, "expected the two left chirps, got {times:?}");
        assert!(found.iter().all(|o| o.spatial_agreement.unwrap() > 0.8), "{found:?}");
        assert!(times.iter().all(|&t| t < 3.0), "{times:?}");

        // Without spatial weighting (or mono) all three are found.
        let blind = SimilarParams {
            spatial_weight: 0.0,
            ..params.clone()
        };
        assert_eq!(find_in_range(&tpl, &mono, &ch, &stft, range.clone(), &blind).len(), 3);
        assert_eq!(find_in_range(&tpl, &mono, &[], &stft, range, &params).len(), 3);
    }

    /// Adds band-limited noise (40 partials between 3 and 6 kHz) gated into
    /// pulses at `rate` Hz (40 % duty, raised-cosine edges) from `start_s`
    /// for `len` seconds; `rate` 0 gives a steady burst.
    fn pulse_burst(out: &mut [f32], sr: u32, start_s: f32, len: f32, rate: f32, amp: f32, seed: u64) {
        let mut rng = seed;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 40) as f32 / (1u64 << 24) as f32
        };
        let partials: Vec<(f32, f32)> = (0..40)
            .map(|_| (3000.0 + 3000.0 * next(), 2.0 * std::f32::consts::PI * next()))
            .collect();
        let start = (start_s * sr as f32) as usize;
        let n = ((len * sr as f32) as usize).min(out.len().saturating_sub(start));
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let gate = if rate > 0.0 {
                let phase = (t * rate).fract();
                if phase < 0.4 {
                    (std::f32::consts::PI * phase / 0.4).sin()
                } else {
                    0.0
                }
            } else {
                1.0
            };
            let v: f32 = partials
                .iter()
                .map(|&(f, ph)| (2.0 * std::f32::consts::PI * f * t + ph).sin())
                .sum();
            out[start + i] += amp * gate * v / 40.0;
        }
    }

    /// Noise plus pulse bursts at 6 Hz: the marked one (0.5 s, 2 s long),
    /// a 20 dB quieter one (4 s, 1.2 s long), a long one under a steady
    /// tone (7 s, 3 s long); decoys in the same band: pulses at 2 Hz (11 s)
    /// and steady noise (13.5 s).
    fn insect_scene(sr: u32) -> Vec<f32> {
        let mut rng = 5u64;
        let mut x: Vec<f32> = (0..16 * sr)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.002
            })
            .collect();
        pulse_burst(&mut x, sr, 0.5, 2.0, 6.0, 0.3, 1);
        pulse_burst(&mut x, sr, 4.0, 1.2, 6.0, 0.03, 2);
        pulse_burst(&mut x, sr, 7.0, 3.0, 6.0, 0.3, 3);
        for i in (7 * sr as usize)..(10 * sr as usize) {
            x[i] += 0.2 * (2.0 * std::f32::consts::PI * 4000.0 * i as f32 / sr as f32).sin();
        }
        pulse_burst(&mut x, sr, 11.0, 2.0, 2.0, 0.3, 4);
        pulse_burst(&mut x, sr, 13.5, 1.5, 0.0, 0.3, 5);
        x
    }

    /// Template of the marked burst (0.5 .. 2.5 s, 2.5 .. 7 kHz) with the
    /// noise measured in the quiet stretch at the end.
    fn insect_template(x: &[f32], stft: &Stft, sr: u32) -> Template {
        let cfg = stft.config();
        let frame = |s: f32| (s * sr as f32 / cfg.hop as f32) as i64;
        let quiet = stft.analyze(x, frame(15.2), 20).power();
        let noise: Vec<f32> = (0..cfg.bins())
            .map(|k| quiet.iter().map(|f| f[k]).sum::<f32>() / quiet.len() as f32)
            .collect();
        let (a, b) = (frame(0.5), frame(2.5));
        let power = stft.analyze(x, a, (b - a) as usize).power();
        let (k0, k1) = (cfg.hz_to_bin(2500.0, sr) as usize, cfg.hz_to_bin(7000.0, sr) as usize);
        let members: Vec<Vec<f32>> = (0..power.len())
            .map(|_| {
                (0..cfg.bins())
                    .map(|k| if (k0..=k1).contains(&k) { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        Template::build(&power, a, &noise, &members).expect("template")
    }

    #[test]
    fn chirps_are_not_taken_for_pulsed_sounds() {
        let sr = 16000;
        let x = scene(sr);
        let stft = Stft::new(StftConfig::default());
        assert!(template(&x, &stft, sr).rhythm.is_none());
    }

    #[test]
    fn pulsed_bursts_are_found_by_rhythm_whatever_their_level_and_length() {
        let sr = 16000;
        let x = insect_scene(sr);
        let stft = Stft::new(StftConfig::default());
        let hop = stft.config().hop as f32;
        let tpl = insect_template(&x, &stft, sr);
        let rhythm = tpl.rhythm.as_ref().expect("pulsed");
        let rate = sr as f32 / hop / rhythm.period;
        assert!((rate - 6.0).abs() < 0.8, "pulse rate {rate}");
        let total = stft.config().frame_count(x.len()) as i64;
        let found = find_in_range(&tpl, &x, &[], &stft, 0..total, &SimilarParams::default());
        let spans: Vec<(f32, f32)> = found
            .iter()
            .map(|o| {
                let s = o.frame as f32 * hop / sr as f32;
                (s, s + o.frames as f32 * hop / sr as f32)
            })
            .collect();
        assert_eq!(found.len(), 3, "{spans:?}");
        for ((s, e), (want_s, want_e)) in spans.iter().zip([(0.5, 2.5), (4.0, 5.2), (7.0, 10.0)]) {
            assert!((s - want_s).abs() < 0.4 && (e - want_e).abs() < 0.4, "{spans:?}");
        }
        assert!(found[0].is_original(&tpl));
        assert!(!found[1].is_original(&tpl) && !found[2].is_original(&tpl));
        // A long burst is applied beyond the template's length.
        let late = found[2].frame + tpl.frames as i64 + 5;
        assert!(found[2].applies_to(&tpl, late));
        assert!(found[2].template_frame(&tpl, late) < tpl.frames);
        // The marked burst only adds frames outside the marked part.
        assert!(!found[0].applies_to(&tpl, tpl.first_frame + 3));
    }

    #[test]
    fn pulsed_bursts_are_the_same_whichever_chunk_asks() {
        let sr = 16000;
        let x = insect_scene(sr);
        let stft = Stft::new(StftConfig::default());
        let tpl = insect_template(&x, &stft, sr);
        let total = stft.config().frame_count(x.len()) as i64;
        let params = SimilarParams::default();
        let whole = find_in_range(&tpl, &x, &[], &stft, 0..total, &params);
        let mut pieced: Vec<Occurrence> = Vec::new();
        for start in (0..total).step_by(16) {
            for o in find_in_range(&tpl, &x, &[], &stft, start..(start + 16).min(total), &params) {
                if !pieced.contains(&o) {
                    pieced.push(o);
                }
            }
        }
        assert_eq!(pieced, whole);
    }

    #[test]
    fn mask_of_a_pulsed_burst_covers_the_band_in_every_frame() {
        let sr = 16000;
        let x = insect_scene(sr);
        let stft = Stft::new(StftConfig::default());
        let tpl = insect_template(&x, &stft, sr);
        let occ = Occurrence {
            frame: 1000,
            frames: 3 * tpl.frames,
            shift: 0,
            score: 1.0,
            spatial_agreement: None,
        };
        let mut power = tpl.noise.clone();
        let k = (tpl.k0 + tpl.k1) / 2;
        power[k] *= 1000.0;
        let mut out = vec![0.0; power.len()];
        // Beyond the template's length, and with membership only in the
        // template's first frame: the band still counts.
        let member = |t: usize, kk: usize| {
            if t == 0 && (tpl.k0..=tpl.k1).contains(&kk) {
                1.0
            } else {
                0.0
            }
        };
        add_occurrence_mask(
            &tpl,
            &occ,
            &SimilarParams::default(),
            1000 + 2 * tpl.frames as i64,
            &power,
            &member,
            &mut out,
        );
        assert!(out[k] > 0.9);
        assert_eq!(out[tpl.k0], 0.0, "no energy there, nothing to take");
    }
}
