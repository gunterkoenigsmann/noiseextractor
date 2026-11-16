//! Sharing a bin between a sound to keep and a sound to remove.
//!
//! When an *extract* filter and a *suppress* filter both claim the same
//! time-frequency bin (the bird's overtone sits on the whistle you want
//! gone), their masks alone cannot say how much of that bin's energy is
//! whose: each mask was computed as if its sound were alone. This module
//! estimates each sound's power in such *contested* bins and gives every
//! filter its share of the bin instead.
//!
//! ## The estimate
//!
//! A marked sound is not just a set of bins; the analysis knows how its
//! parts relate (see [`SourceModel`]): how loud each harmonic partial is
//! relative to the fundamental along the drifting pitch, and how loud each
//! co-modulated band is relative to it. So in every frame:
//!
//! 1. the sound's expected *shape* `S(k)` is predicted from the pitch track
//!    (moved to the place of a repetition, if this is one), the partial
//!    levels and the window's main lobe;
//! 2. its loudness `a` in this frame is measured on the bins of the shape
//!    that nobody else claims (the robust median of `(P − N) / S`): the
//!    fundamental and the overtones that are clear tell how strong the
//!    overtone in the collision should be;
//! 3. where every bin of the sound is contested in a frame, the loudness is
//!    taken from the nearest frame (within [`FALLBACK_FRAMES`]) where it
//!    could be measured;
//! 4. in a contested bin the sound is expected to have `p = a·S(k)`. Sounds
//!    without a model (a noise burst has no pitch) get what the modelled
//!    sounds and the background noise leave over.
//!
//! In multi-channel recordings each sound's expected power in a contested
//! bin is also weighted by how well the bin's level ratio and phase between
//! the channels fit the sound's direction (see [`crate::spatial`]): energy
//! that arrives from the bird's place is more likely the bird's.
//!
//! Each filter's mask in the bin becomes `√(p / P)`, the amplitude share of
//! its power. The extract filters then keep exactly their sounds' shares,
//! so the suppress masks are not applied a second time there.

use crate::analysis::RegionAnalysis;
use crate::spatial::Signature;
use crate::stats::quantile;
use crate::stft::Spectra;
use realfft::num_complex::Complex32;

/// Mask value from which a filter counts as claiming a bin.
pub const CLAIM: f32 = 0.2;
/// How many frames away the loudness of a sound may be borrowed from when
/// it cannot be measured in a frame.
pub const FALLBACK_FRAMES: usize = 8;
/// Smallest weight a direction mismatch can give (so a slightly wrong
/// direction estimate never zeroes a sound completely).
const MIN_DIRECTION_WEIGHT: f32 = 0.05;
/// Bins of a shape below this fraction of its maximum are not used to
/// measure the loudness (only the clear peaks are).
const PEAK_FRACTION: f32 = 0.1;

/// How the parts of a marked sound relate in level, learned from the
/// marked instance.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceModel {
    /// Harmonic partials: frequency ratio to the tracked pitch and power
    /// relative to the fundamental's peak.
    pub partials: Vec<(f32, f32)>,
    /// Co-modulated bins: bin index and power relative to the fundamental.
    pub bins: Vec<(usize, f32)>,
}

/// Power response of a Hann window's main lobe `d` bins away from a
/// sinusoid (1 at the peak, ¼ one bin away, 0 from two bins on).
pub fn hann_lobe(d: f32) -> f32 {
    let d = d.abs();
    if d >= 2.0 {
        return 0.0;
    }
    if (d - 1.0).abs() < 1e-4 {
        return 0.25;
    }
    let w = if d < 1e-6 {
        1.0
    } else {
        let x = std::f32::consts::PI * d;
        x.sin() / x / (1.0 - d * d)
    };
    w * w
}

/// Net power (above the noise) of the strongest bin within ±`tol` bins of
/// `centre`, or `None` outside the spectrum.
fn peak_net(power: &[f32], noise: &[f32], centre: f32, tol: f32) -> Option<f32> {
    let bins = power.len();
    if centre < 0.0 || centre > (bins - 1) as f32 {
        return None;
    }
    let lo = (centre - tol).floor().max(0.0) as usize;
    let hi = ((centre + tol).ceil() as usize).min(bins - 1);
    Some((lo..=hi).map(|k| (power[k] - noise[k]).max(0.0)).fold(0.0, f32::max))
}

impl SourceModel {
    /// Learns the level relations from the analysed instance: `power` are
    /// the analysed frames (the analysis' first frame first), `noise` the
    /// noise power per bin. Partials whose harmonic mask along the track is
    /// weak (e.g. rejected for coming from another direction) are left
    /// out. Returns `None` without a pitch track (nothing to relate to).
    pub fn build(power: &[Vec<f32>], noise: &[f32], a: &RegionAnalysis, tolerance: f32) -> Option<Self> {
        let frames: Vec<(usize, f32, f32)> = (0..a.frames.min(power.len()))
            .filter_map(|t| {
                let f0 = a.pitch_track[t]?;
                let e = peak_net(&power[t], noise, f0, 1.0)?;
                (e > 0.0).then_some((t, f0, e))
            })
            .collect();
        if frames.len() < 3 {
            return None;
        }
        let mut model = SourceModel::default();
        for p in &a.partials {
            let (mut levels, mut membership) = (Vec::new(), Vec::new());
            for &(t, f0, e) in &frames {
                let centre = f0 * p.ratio;
                let k = centre.round() as usize;
                if k >= a.bins {
                    continue;
                }
                membership.push(a.harmonics[t][k]);
                if a.harmonics[t][k] >= 0.3 {
                    if let Some(v) = peak_net(&power[t], noise, centre, (centre * tolerance).max(1.0)) {
                        levels.push(v / e);
                    }
                }
            }
            let strong = !membership.is_empty() && membership.iter().sum::<f32>() / membership.len() as f32 >= 0.3;
            if strong && levels.len() >= 3 {
                model.partials.push((p.ratio, quantile(&levels, 0.5)));
            }
        }
        for k in 0..a.bins {
            if (0..a.frames).all(|t| a.comodulated[t][k] < 0.3) {
                continue;
            }
            let levels: Vec<f32> = frames
                .iter()
                .map(|&(t, _, e)| (power[t][k] - noise[k]).max(0.0) / e)
                .collect();
            model.bins.push((k, quantile(&levels, 0.5)));
        }
        Some(model)
    }

    /// Predicted relative power of every bin (`out`, overwritten) for a
    /// frame whose pitch is `pitch` (bins, `None` where the sound is
    /// silent), moved by `shift` bins. The fundamental has level 1.
    pub fn shape(&self, pitch: Option<f32>, shift: f32, out: &mut [f32]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        let bins = out.len();
        let mut lobe = |centre: f32, level: f32| {
            if !(0.0..bins as f32).contains(&centre) {
                return;
            }
            let lo = (centre - 2.0).ceil().max(0.0) as usize;
            let hi = ((centre + 2.0).floor() as usize).min(bins - 1);
            for k in lo..=hi {
                out[k] = out[k].max(level * hann_lobe(k as f32 - centre));
            }
        };
        if let Some(f0) = pitch {
            lobe(f0 + shift, 1.0);
            for &(ratio, level) in &self.partials {
                lobe(f0 * ratio + shift, level);
            }
        }
        for &(k, level) in &self.bins {
            let k = k as f32 + shift;
            if (0.0..bins as f32).contains(&k) {
                let k = k as usize;
                out[k] = out[k].max(level);
            }
        }
    }
}

/// One appearance of a filter's sound in a frame: the original or a
/// repetition, as `(analysis frame index, frequency shift in bins)`.
pub type Placement = (usize, f32);

/// What [`share_frame`] needs to know about one event filter in one frame.
pub struct Claimant<'a> {
    /// Whether the filter extracts (else it suppresses).
    pub extract: bool,
    /// The filter's mask in this frame; replaced by its shares in
    /// contested bins.
    pub mask: &'a mut [f32],
    /// The sound's predicted relative power in this frame (sum over its
    /// placements), or `None` without a model.
    pub shape: Option<Vec<f32>>,
    /// The sound's loudness in this frame (see [`loudness`]), if known.
    pub loudness: Option<f32>,
    /// Where the sound comes from, with the level and delay tolerances
    /// (multi-channel recordings only).
    pub direction: Option<(&'a Signature, f32, f32)>,
}

/// Bins of a frame that an extract and a suppress filter both claim, from
/// the filters' masks as `(extract?, mask)`.
pub fn contested(masks: &[(bool, &[f32])]) -> Vec<bool> {
    let bins = masks.first().map_or(0, |(_, m)| m.len());
    (0..bins)
        .map(|k| {
            let claims = |extract: bool| masks.iter().any(|(x, m)| *x == extract && m[k] >= CLAIM);
            claims(true) && claims(false)
        })
        .collect()
}

/// Loudness of a sound in a frame: the median of `(P − N) / S` over the
/// clear peaks of its shape `S` that are not `contested`. `None` if there
/// is no such bin.
pub fn loudness(shape: &[f32], power: &[f32], noise: &[f32], contested: &[bool]) -> Option<f32> {
    let max = shape.iter().cloned().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return None;
    }
    let ratios: Vec<f32> = (0..shape.len())
        .filter(|&k| shape[k] >= PEAK_FRACTION * max && !contested[k])
        .map(|k| (power[k] - noise[k]).max(0.0) / shape[k])
        .collect();
    (!ratios.is_empty()).then(|| quantile(&ratios, 0.5))
}

/// Fills frames without a loudness from the nearest frame (within
/// [`FALLBACK_FRAMES`]) that has one; ties go to the earlier frame.
pub fn fill_gaps(loudness: &mut [Option<f32>]) {
    let known = loudness.to_vec();
    for (t, v) in loudness.iter_mut().enumerate() {
        if v.is_some() {
            continue;
        }
        *v = (1..=FALLBACK_FRAMES).find_map(|d| {
            t.checked_sub(d)
                .and_then(|i| known[i])
                .or_else(|| known.get(t + d).copied().flatten())
        });
    }
}

/// How well bin `k` of frame `t` fits `direction` (1 = exactly), from the
/// channels' spectra (`channels[c].frames[t][k]`).
fn direction_weight(
    channels: &[Spectra],
    t: usize,
    k: usize,
    fft_size: usize,
    direction: (&Signature, f32, f32),
) -> f32 {
    let (sig, level_tol, delay_tol) = direction;
    let a = channels[0].frames[t][k];
    let mut weight = 1.0f32;
    for (c, cue) in channels[1..].iter().zip(&sig.pairs) {
        let b = c.frames[t][k];
        let (pa, pb) = (a.norm_sqr(), b.norm_sqr());
        if pa <= 1e-20 || pb <= 1e-20 {
            continue;
        }
        let dl = (10.0 * (pb / pa).log10() - cue.level_db) / level_tol.max(1e-3);
        let dd = if k == 0 {
            0.0
        } else {
            let expected = 2.0 * std::f32::consts::PI * k as f32 * cue.delay_samples / fft_size as f32;
            let err = (a * b.conj() * Complex32::from_polar(1.0, -expected)).arg();
            err * fft_size as f32 / (2.0 * std::f32::consts::PI * k as f32) / delay_tol.max(1e-3)
        };
        weight *= (-0.5 * (dl * dl + dd * dd)).exp();
    }
    weight.max(MIN_DIRECTION_WEIGHT)
}

/// Replaces the claimants' masks in the `contested` bins of frame `t` by
/// their shares of the bin's power `power` (minus the background `noise`),
/// as described in the module documentation. `channels` are the channel
/// spectra for direction weighting (empty for mono or when unknown).
#[allow(clippy::too_many_arguments)]
pub fn share_frame(
    claimants: &mut [Claimant],
    contested: &[bool],
    power: &[f32],
    noise: &[f32],
    channels: &[Spectra],
    t: usize,
    fft_size: usize,
) {
    let mut expected = vec![0.0f32; claimants.len()];
    for k in (0..power.len()).filter(|&k| contested[k]) {
        let p = power[k].max(1e-20);
        let available = (power[k] - noise[k]).max(0.0);
        // Modelled sounds first: loudness × shape, weighted by direction.
        let mut explained = 0.0f32;
        let mut leftover_weight = 0.0f32;
        for (i, c) in claimants.iter().enumerate() {
            expected[i] = 0.0;
            if c.mask[k] <= 0.0 {
                continue;
            }
            let dir = match (c.direction, channels.len() >= 2) {
                (Some(d), true) => direction_weight(channels, t, k, fft_size, d),
                _ => 1.0,
            };
            match (&c.shape, c.loudness) {
                (Some(shape), Some(a)) => {
                    expected[i] = a * shape[k] * dir;
                    explained += expected[i];
                }
                // Unmodelled: weight for the leftover, kept in `expected`
                // as a negative number until the leftover is known.
                _ => {
                    expected[i] = -(c.mask[k] * dir);
                    leftover_weight += c.mask[k] * dir;
                }
            }
        }
        // More predicted than there is: scale the predictions down.
        let scale = if explained > available && explained > 0.0 {
            available / explained
        } else {
            1.0
        };
        let leftover = (available - explained * scale).max(0.0);
        for (i, c) in claimants.iter_mut().enumerate() {
            if c.mask[k] <= 0.0 {
                continue;
            }
            let share = if expected[i] >= 0.0 {
                expected[i] * scale
            } else if leftover_weight > 0.0 {
                leftover * -expected[i] / leftover_weight
            } else {
                0.0
            };
            c.mask[k] = (share / p).clamp(0.0, 1.0).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_lobe_has_the_known_values() {
        assert!((hann_lobe(0.0) - 1.0).abs() < 1e-6);
        assert!((hann_lobe(1.0) - 0.25).abs() < 1e-6);
        assert!((hann_lobe(-1.0) - 0.25).abs() < 1e-6);
        assert_eq!(hann_lobe(2.0), 0.0);
        // Half a bin off: (sin(π/2)/(π/2)/(3/4))² = (8/(3π))².
        let half = (8.0 / (3.0 * std::f32::consts::PI)).powi(2);
        assert!((hann_lobe(0.5) - half).abs() < 1e-5);
    }

    #[test]
    fn shape_places_partials_along_the_pitch() {
        let m = SourceModel {
            partials: vec![(2.0, 0.5)],
            bins: vec![(40, 0.1)],
        };
        let mut out = vec![0.0; 64];
        m.shape(Some(10.0), 0.0, &mut out);
        assert_eq!(out[10], 1.0);
        assert_eq!(out[20], 0.5);
        assert!((out[21] - 0.125).abs() < 1e-6);
        assert_eq!(out[40], 0.1);
        // A repetition two bins higher.
        m.shape(Some(10.0), 2.0, &mut out);
        assert_eq!((out[12], out[22], out[42]), (1.0, 0.5, 0.1));
        // Silent frame: only the co-modulated bins.
        m.shape(None, 0.0, &mut out);
        assert_eq!(out.iter().filter(|&&v| v > 0.0).count(), 1);
    }

    #[test]
    fn loudness_uses_clear_uncontested_peaks_only() {
        let shape = [1.0, 0.5, 0.0, 0.01];
        let power = [11.0, 100.0, 50.0, 7.0];
        let noise = [1.0; 4];
        // Bin 1 is contested (an intruder makes it far too loud), bin 3 is
        // too weak a part of the shape to count.
        assert_eq!(
            loudness(&shape, &power, &noise, &[false, true, false, false]),
            Some(10.0)
        );
        assert_eq!(loudness(&shape, &power, &noise, &[true, true, true, true]), None);
    }

    #[test]
    fn gaps_are_filled_from_the_nearest_measured_frame() {
        let mut l = vec![Some(1.0), None, None, Some(4.0), None];
        fill_gaps(&mut l);
        assert_eq!(l, vec![Some(1.0), Some(1.0), Some(4.0), Some(4.0), Some(4.0)]);
        let mut far = vec![None; FALLBACK_FRAMES + 2];
        far[0] = Some(2.0);
        fill_gaps(&mut far);
        assert_eq!(far[FALLBACK_FRAMES], Some(2.0));
        assert_eq!(far[FALLBACK_FRAMES + 1], None);
    }

    #[test]
    fn a_modelled_sound_gets_its_predicted_share_the_other_the_rest() {
        // One bin with power 10 over noise 1: the bird is predicted at 3,
        // the whistle has no model and gets the remaining 6.
        let (mut bird, mut whistle) = (vec![0.9f32], vec![0.9f32]);
        let mut claimants = [
            Claimant {
                extract: true,
                mask: &mut bird,
                shape: Some(vec![0.5]),
                loudness: Some(6.0),
                direction: None,
            },
            Claimant {
                extract: false,
                mask: &mut whistle,
                shape: None,
                loudness: None,
                direction: None,
            },
        ];
        let contested = contested(&[(true, &[0.9]), (false, &[0.9])]);
        assert_eq!(contested, vec![true]);
        assert_eq!(super::contested(&[(true, &[0.9]), (true, &[0.9])]), vec![false]);
        share_frame(&mut claimants, &contested, &[10.0], &[1.0], &[], 0, 64);
        assert!((bird[0] - 0.3f32.sqrt()).abs() < 1e-6, "{bird:?}");
        assert!((whistle[0] - 0.6f32.sqrt()).abs() < 1e-6, "{whistle:?}");
    }

    #[test]
    fn predictions_beyond_the_bin_are_scaled_down() {
        let (mut a, mut b) = (vec![1.0f32], vec![1.0f32]);
        let mut claimants = [
            Claimant {
                extract: true,
                mask: &mut a,
                shape: Some(vec![1.0]),
                loudness: Some(9.0),
                direction: None,
            },
            Claimant {
                extract: false,
                mask: &mut b,
                shape: Some(vec![1.0]),
                loudness: Some(9.0),
                direction: None,
            },
        ];
        share_frame(&mut claimants, &[true], &[10.0], &[0.0], &[], 0, 64);
        // 18 predicted, 10 there: half each.
        assert!((a[0] - 0.5f32.sqrt()).abs() < 1e-6 && (b[0] - 0.5f32.sqrt()).abs() < 1e-6);
    }
}
