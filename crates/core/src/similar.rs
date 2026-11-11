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

use crate::stft::Stft;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Tuning of the search for similar occurrences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimilarParams {
    /// Correlation (0..1) above which a place counts as another occurrence.
    pub threshold: f32,
    /// Largest frequency shift in bins tried when matching; small values
    /// tolerate slightly higher or lower repetitions.
    pub max_shift_bins: usize,
}

impl Default for SimilarParams {
    /// Threshold 0.6 and ±2 bins (≈ ±47 Hz at 48 kHz with 2048-point FFTs).
    fn default() -> Self {
        Self {
            threshold: 0.6,
            max_shift_bins: 2,
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
}

/// One place where the template matched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Occurrence {
    /// Global frame where the matched patch starts.
    pub frame: i64,
    /// Frequency shift in bins relative to the template.
    pub shift: i32,
    /// Correlation with the template (0..1).
    pub score: f32,
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
        })
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
pub fn find_in_range(
    template: &Template,
    mono: &[f32],
    stft: &Stft,
    range: Range<i64>,
    params: &SimilarParams,
) -> Vec<Occurrence> {
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
    let scores: Vec<(f32, i32)> = (scored.start..scored.end)
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

    let at = |o: i64| scores[(o - scored.start) as usize];
    let mut out = Vec::new();
    for o in cand {
        let (score, shift) = at(o);
        if score < params.threshold {
            continue;
        }
        let lo = (o - radius).max(scored.start);
        let hi = (o + radius).min(scored.end - 1);
        // Strictly greater than earlier neighbours, at least as great as
        // later ones: exactly one offset of a plateau wins.
        let is_peak = (lo..o).all(|p| at(p).0 < score) && (o + 1..=hi).all(|p| at(p).0 <= score);
        if is_peak {
            out.push(Occurrence { frame: o, shift, score });
        }
    }
    out
}

impl Occurrence {
    /// Whether this is the marked sound itself (the template matches its
    /// own position, give or take a few frames).
    pub fn is_original(&self, template: &Template) -> bool {
        (self.frame - template.first_frame).abs() <= (template.frames as i64 / 2).max(1)
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
    if rel < 0 || rel as usize >= template.frames {
        return;
    }
    let weight = occurrence_weight(occ.score, params);
    for k in template.k0..=template.k1 {
        let m = member(rel as usize, k);
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
        let whole = find_in_range(&tpl, &x, &stft, 0..total, &p);
        let mut pieces = Vec::new();
        let mut a = 0;
        while a < total {
            let b = (a + 37).min(total);
            pieces.extend(find_in_range(&tpl, &x, &stft, a..b, &p));
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
        let found = find_in_range(&tpl, &x, &stft, 0..20, &SimilarParams::default());
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
            shift: 1,
            score: 1.0,
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
}
