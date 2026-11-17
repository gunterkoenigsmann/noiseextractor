//! Recognising pulsed sounds (crickets, katydids, trills, machinery) by
//! their rhythm instead of their exact spectro-temporal shape.
//!
//! The 2-D template of [`crate::similar`] needs every cell of a repetition
//! to line up with the marked sound. A pulse train defeats that: the
//! pulses drift against each other, bursts differ in length, and a more
//! distant burst sinks partly into the noise floor, which flattens the log
//! levels the correlation looks at. What stays the same from burst to
//! burst is
//!
//! * the **rhythm**: the spectrum of the band's loudness over time, whose
//!   magnitude does not depend on where a pulse starts, and
//! * the **band shape**: which parts of the band are busy, measured as
//!   level above the noise floor with the mean removed, so a quieter burst
//!   has the same shape.
//!
//! Loudness only has to reach a minimum. Everything is measured in STFT
//! frames, so no sample rate is needed.

/// Number of sub-bands the template's frequency range is split into for
/// the band shape (fewer if the range has fewer bins).
const BANDS: usize = 16;
/// Shortest template (in frames) that is checked for a rhythm.
const MIN_FRAMES: usize = 16;
/// Pulses a window must hold at least, so one hump does not count as a
/// rhythm. The lowest modulation bin used is this one.
const MIN_PULSES: usize = 3;
/// Pulses a matching window should hold; longer windows give a sharper
/// rhythm, shorter ones follow short bursts.
const WINDOW_PULSES: f32 = 4.0;
/// Smallest RMS swing (dB) of the loudness within a window for it to count
/// as pulsing at all; steady noise has less.
const MIN_SWING_DB: f32 = 1.0;
/// Share of the loudness swing that must lie at the strongest pulse rate
/// (± one bin) for the marked sound to count as pulsed.
const MIN_PEAKINESS: f32 = 0.2;

/// The rhythm and band shape of a pulsed sound.
#[derive(Clone, Debug, PartialEq)]
pub struct Rhythm {
    /// Length of the matching window in frames.
    pub window: usize,
    /// Pulse period in frames.
    pub period: f32,
    /// Sub-band edges (absolute bins), `bands + 1` values; band `i` is
    /// `edges[i] .. edges[i + 1]`.
    edges: Vec<usize>,
    /// Unit vector of modulation powers for bins `1 ..= window / 2` of a
    /// window.
    modulation: Vec<f32>,
    /// Unit vector of the mean-free band shape.
    shape: Vec<f32>,
}

/// What one window of the recording looks like.
struct WindowFeatures {
    /// Unit modulation vector (zero if there is no swing).
    modulation: Vec<f32>,
    /// Unit, mean-free band shape (zero if flat).
    shape: Vec<f32>,
    /// Mean loudness above the noise floor (dB).
    level: f32,
    /// RMS swing of the loudness (dB).
    swing: f32,
}

/// Scales `v` to unit length; leaves an all-zero vector alone.
fn normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Dot product.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Splits bins `k0 ..= k1` into at most [`BANDS`] contiguous bands.
fn band_edges(k0: usize, k1: usize) -> Vec<usize> {
    let width = k1 + 1 - k0;
    let bands = BANDS.min(width);
    (0..=bands).map(|i| k0 + i * width / bands).collect()
}

/// Level above the noise floor (dB, at least 0) of each band of one frame,
/// averaged over the band's bins.
pub(crate) fn band_levels(frame: &[f32], noise: &[f32], edges: &[usize]) -> Vec<f32> {
    edges
        .windows(2)
        .map(|e| {
            let bins = e[0]..e[1];
            let n = bins.len() as f32;
            bins.map(|k| (10.0 * (frame[k] / noise[k].max(1e-20) + 1e-12).log10()).max(0.0))
                .sum::<f32>()
                / n
        })
        .collect()
}

/// Magnitudes of the Hann-windowed, mean-free `env` at DFT bins `m_lo ..=
/// m_hi` (direct DFT; windows are a few dozen frames).
fn modulation_spectrum(env: &[f32], m_lo: usize, m_hi: usize) -> Vec<f32> {
    let n = env.len();
    let mean = env.iter().sum::<f32>() / n as f32;
    let tapered: Vec<f32> = env
        .iter()
        .enumerate()
        .map(|(i, &e)| {
            let hann = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (i as f32 + 0.5) / n as f32).cos();
            (e - mean) * hann
        })
        .collect();
    (m_lo..=m_hi)
        .map(|m| {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (i, &x) in tapered.iter().enumerate() {
                let ph = 2.0 * std::f32::consts::PI * (m * i) as f32 / n as f32;
                re += x * ph.cos();
                im -= x * ph.sin();
            }
            (re * re + im * im).sqrt()
        })
        .collect()
}

/// Mean and RMS deviation of `v`.
fn mean_and_swing(v: &[f32]) -> (f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Features of the `window` frames of `levels` (per-frame band levels)
/// starting at row `start`.
fn window_features(levels: &[Vec<f32>], start: usize, window: usize) -> WindowFeatures {
    let rows = &levels[start..start + window];
    let env: Vec<f32> = rows.iter().map(|r| r.iter().sum::<f32>() / r.len() as f32).collect();
    let (level, swing) = mean_and_swing(&env);
    // Power, not magnitude: a clear rhythm is one dominant bin, and squaring
    // keeps a broad spectrum (steady noise, a slow hump) from resembling it.
    // The slow bins are included so energy there counts against a match.
    let mut modulation: Vec<f32> = modulation_spectrum(&env, 1, window / 2).iter().map(|a| a * a).collect();
    normalise(&mut modulation);
    let bands = rows[0].len();
    let mut shape: Vec<f32> = (0..bands)
        .map(|b| rows.iter().map(|r| r[b]).sum::<f32>() / window as f32)
        .collect();
    let m = shape.iter().sum::<f32>() / bands as f32;
    shape.iter_mut().for_each(|s| *s -= m);
    normalise(&mut shape);
    WindowFeatures {
        modulation,
        shape,
        level,
        swing,
    }
}

impl Rhythm {
    /// Checks whether the marked sound (`power` rows of the template,
    /// bins `k0 ..= k1`, noise power per bin `noise`) pulses at a clear
    /// rate, and if so describes its rhythm and band shape. Returns `None`
    /// for sounds without a rhythm (a single chirp, a steady tone), which
    /// keep using the 2-D template.
    pub fn build(power: &[Vec<f32>], noise: &[f32], k0: usize, k1: usize) -> Option<Self> {
        let frames = power.len();
        if frames < MIN_FRAMES || k1 < k0 {
            return None;
        }
        let edges = band_edges(k0, k1);
        let levels: Vec<Vec<f32>> = power.iter().map(|f| band_levels(f, noise, &edges)).collect();
        let env: Vec<f32> = levels.iter().map(|r| r.iter().sum::<f32>() / r.len() as f32).collect();
        let (_, swing) = mean_and_swing(&env);
        if swing < MIN_SWING_DB {
            return None;
        }
        // Pulse rate: the strongest modulation with at least MIN_PULSES
        // periods in the template, compared with all modulation (including
        // the slow rise and fall of the burst as a whole).
        let all = modulation_spectrum(&env, 1, frames / 2);
        let energy: f32 = all.iter().map(|a| a * a).sum();
        let (peak, _) = all
            .iter()
            .enumerate()
            .skip(MIN_PULSES - 1)
            .fold((0usize, -1.0f32), |b, (i, &a)| if a > b.1 { (i, a) } else { b });
        if peak < MIN_PULSES - 1 {
            return None;
        }
        let around: f32 = all[peak.saturating_sub(1)..(peak + 2).min(all.len())]
            .iter()
            .map(|a| a * a)
            .sum();
        if energy <= 0.0 || around / energy < MIN_PEAKINESS {
            return None;
        }
        let period = frames as f32 / (peak + 1) as f32;
        let window = ((WINDOW_PULSES * period).round() as usize).clamp(MIN_FRAMES, frames);
        // Average the features of the template's own windows.
        let step = (window / 4).max(1);
        let mut modulation = vec![0.0f32; window / 2];
        let mut shape = vec![0.0f32; edges.len() - 1];
        let mut used = 0;
        for start in (0..=frames - window).step_by(step) {
            let f = window_features(&levels, start, window);
            if f.swing < MIN_SWING_DB {
                continue;
            }
            modulation.iter_mut().zip(&f.modulation).for_each(|(a, b)| *a += b);
            shape.iter_mut().zip(&f.shape).for_each(|(a, b)| *a += b);
            used += 1;
        }
        if used == 0 {
            return None;
        }
        normalise(&mut modulation);
        normalise(&mut shape);
        Some(Self {
            window,
            period,
            edges,
            modulation,
            shape,
        })
    }

    /// Band levels of each power frame, as [`Rhythm::score`] needs them.
    pub fn levels(&self, power: &[Vec<f32>], noise: &[f32]) -> Vec<Vec<f32>> {
        power.iter().map(|f| band_levels(f, noise, &self.edges)).collect()
    }

    /// How well the window of `levels` starting at row `start` matches
    /// (0..1): rhythm similarity times the square root of the band-shape
    /// similarity. 0 when the window is quieter than `min_level_db` above
    /// the noise floor or does not pulse at all.
    pub fn score(&self, levels: &[Vec<f32>], start: usize, min_level_db: f32) -> f32 {
        let f = window_features(levels, start, self.window);
        if f.level < min_level_db || f.swing < MIN_SWING_DB {
            return 0.0;
        }
        dot(&f.modulation, &self.modulation).max(0.0) * dot(&f.shape, &self.shape).max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Power frames (`frames` × `bins`) with noise 1 everywhere and, from
    /// frame `on` to `off`, pulses of `gain` in bins `lo..hi` lasting
    /// `duty` of every `period` frames.
    #[allow(clippy::too_many_arguments)]
    fn pulses(
        frames: usize,
        bins: usize,
        on: usize,
        off: usize,
        lo: usize,
        hi: usize,
        period: f32,
        duty: f32,
        gain: f32,
    ) -> Vec<Vec<f32>> {
        (0..frames)
            .map(|t| {
                let pulsing = (on..off).contains(&t) && ((t - on) as f32 % period) < duty * period;
                (0..bins)
                    .map(|k| {
                        if pulsing && (lo..hi).contains(&k) {
                            1.0 + gain
                        } else {
                            1.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn pulse_train_has_a_rhythm_with_the_right_period() {
        let p = pulses(120, 64, 0, 120, 20, 50, 10.0, 0.4, 1000.0);
        let r = Rhythm::build(&p, &[1.0; 64], 18, 52).expect("pulsed");
        assert!((r.period - 10.0).abs() < 1.0, "period {}", r.period);
        assert_eq!(r.window, 40);
    }

    #[test]
    fn a_single_hump_or_a_steady_tone_has_no_rhythm() {
        let hump = pulses(120, 64, 30, 90, 20, 50, 1000.0, 1.0, 1000.0);
        assert!(Rhythm::build(&hump, &[1.0; 64], 18, 52).is_none());
        let steady = pulses(120, 64, 0, 120, 20, 50, 1000.0, 1.0, 1000.0);
        assert!(Rhythm::build(&steady, &[1.0; 64], 18, 52).is_none());
    }

    #[test]
    fn quieter_and_shifted_repetition_still_matches() {
        let noise = [1.0; 64];
        let p = pulses(120, 64, 0, 120, 20, 50, 10.0, 0.4, 1000.0);
        let r = Rhythm::build(&p, &noise, 18, 52).expect("pulsed");
        // 20 dB quieter, pulses starting at another phase.
        let q = pulses(120, 64, 3, 120, 20, 50, 10.0, 0.4, 10.0);
        let levels = r.levels(&q, &noise);
        assert!(r.score(&levels, 10, 3.0) > 0.9);
        // Below the minimum level it is ignored.
        assert_eq!(r.score(&levels, 10, 15.0), 0.0);
    }

    #[test]
    fn other_rhythm_or_other_band_does_not_match() {
        let noise = [1.0; 64];
        let p = pulses(120, 64, 0, 120, 20, 50, 10.0, 0.4, 1000.0);
        let r = Rhythm::build(&p, &noise, 18, 52).expect("pulsed");
        let faster = pulses(120, 64, 0, 120, 20, 50, 4.0, 0.5, 1000.0);
        assert!(r.score(&r.levels(&faster, &noise), 10, 3.0) < 0.4);
        let elsewhere = pulses(120, 64, 0, 120, 18, 26, 10.0, 0.4, 1000.0);
        assert!(r.score(&r.levels(&elsewhere, &noise), 10, 3.0) < 0.6);
    }
}
