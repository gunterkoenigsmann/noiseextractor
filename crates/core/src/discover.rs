//! Finding out automatically which sounds a recording is made of.
//!
//! ## Idea
//!
//! A recording that is not too complex consists of a handful of sounds that
//! come back again and again: a frog's croak, a bird's call, the hum of a
//! road. Each of them has a typical spectrum, and only its loudness changes
//! over time. Non-negative matrix factorisation (NMF) finds exactly such a
//! description: it writes the magnitude spectrogram `V` (bins × frames) as
//!
//! ```text
//!   V ≈ W · H        W: one spectral *pattern* per column (bins × C)
//!                    H: each pattern's loudness per frame   (C × frames)
//! ```
//!
//! with every entry non-negative, so patterns can only add up, never
//! cancel. The number of patterns `C` is the user's choice. The background
//! noise gets a pattern of its own whose spectrum is fixed to the noise
//! estimate, so the learned patterns describe the sounds, not the hiss.
//!
//! The patterns are learned with Lee & Seung's multiplicative updates for
//! the (generalised) Kullback–Leibler divergence, which suits audio
//! magnitudes better than the squared error: quiet details still count.
//!
//! ## From patterns to filters
//!
//! While processing, each frame's loudnesses `h` are found anew with the
//! patterns fixed (a small non-negative fit per frame, independent of the
//! neighbouring frames, so chunked processing stays seamless). A filter that
//! owns the set `S` of patterns then gets the Wiener-like mask
//!
//! ```text
//!   m(k) = √( Σ_{c∈S} (W_kc·h_c)²  /  Σ_{all c} (W_kc·h_c)² )
//! ```
//!
//! i.e. the amplitude share of the bin's power its patterns explain, which
//! fits the soft-mask model of [`crate::filter`]: suppress and extract are
//! power complements, and the expected power `Σ_{c∈S} (W_kc·h_c)²` takes part
//! in sharing contested bins (see [`crate::share`]).
//!
//! ## Grouping patterns into sounds
//!
//! One sound often needs several patterns (a call with two notes, a voice
//! with changing vowels). Patterns are suggested as one filter when their
//! loudness rises and falls together on the scale of half a second and, in
//! multi-channel recordings, when they come from the same direction
//! (level difference and delay between the channels, see
//! [`crate::spatial`]). The user can still merge, split, rename or discard
//! the suggestions.

use crate::noise::{estimate_noise, NoiseEstimate};
use crate::spatial::{agreement, from_sums, PairSums, Signature};
use crate::stats::pearson;
use crate::stft::{Stft, StftConfig};
use crate::AudioData;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Frames per contiguous block when only part of a long recording is used
/// to learn the patterns. Contiguous blocks keep the loudness curves
/// meaningful for grouping.
const BLOCK_FRAMES: usize = 64;
/// Multiplicative updates per frame when the loudnesses are fitted during
/// processing. They converge fast because only a handful of numbers are
/// fitted; 40 leaves the masks stable to well below 0.1 dB.
pub const INFER_ITERATIONS: usize = 40;
/// Added to model values before dividing, so silent bins do not divide by 0.
const EPS: f32 = 1e-12;
/// Time scale (seconds) on which the loudness of patterns of one sound must
/// rise and fall together to be grouped.
const GROUP_SMOOTH_S: f32 = 0.5;

/// Settings of the search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DiscoverParams {
    /// Number of spectral patterns to learn (besides the background noise).
    pub components: usize,
    /// Learning iterations.
    pub iterations: usize,
    /// At most this many frames are used for learning; longer recordings
    /// (or ranges) are sampled in evenly spread blocks.
    pub max_frames: usize,
    /// Only this time range (seconds) is used for learning; the patterns
    /// apply to the whole recording anyway.
    pub range_s: Option<(f64, f64)>,
    /// Suggest patterns that belong together as one filter.
    pub group: bool,
    /// Smallest correlation of the (smoothed) loudness curves for grouping
    /// in multi-channel recordings, where the direction must agree as well.
    pub min_correlation_stereo: f32,
    /// Smallest correlation of the loudness curves for grouping in mono
    /// recordings, where timing is the only evidence (stricter).
    pub min_correlation_mono: f32,
    /// Level tolerance (dB) for "the same direction".
    pub level_tolerance_db: f32,
    /// Delay tolerance (samples) for "the same direction", on top of each
    /// direction's own measurement uncertainty.
    pub delay_tolerance_samples: f32,
    /// Largest inter-channel delay searched, in samples.
    pub max_delay_samples: f32,
}

impl Default for DiscoverParams {
    fn default() -> Self {
        Self {
            components: 6,
            iterations: 120,
            max_frames: 2500,
            range_s: None,
            group: true,
            min_correlation_stereo: 0.5,
            min_correlation_mono: 0.75,
            level_tolerance_db: 3.0,
            delay_tolerance_samples: 2.0,
            max_delay_samples: 64.0,
        }
    }
}

/// The learned spectral patterns, saved in the project and shared by all
/// filters suggested from one search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decomposition {
    /// FFT size the patterns were learned with; they only apply to the
    /// same STFT.
    pub fft_size: usize,
    /// One magnitude pattern per component (each sums to 1), strongest
    /// first.
    pub patterns: Vec<Vec<f32>>,
    /// Fixed magnitude pattern of the background noise (sums to 1); empty
    /// if none. It takes part in every fit but belongs to no filter.
    #[serde(default)]
    pub background: Vec<f32>,
    /// Where each component comes from (multi-channel recordings only).
    #[serde(default)]
    pub directions: Vec<Option<Signature>>,
    /// Human-readable description of each component.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Level tolerance (dB) used when sharing bins by direction.
    pub level_tolerance_db: f32,
    /// Delay tolerance (samples) used when sharing bins by direction.
    pub delay_tolerance_samples: f32,
}

/// Loudness of every pattern (the background last, if any) in one frame,
/// as fitted by [`Decomposition::fit_frame`].
#[derive(Clone, Debug, PartialEq)]
pub struct FrameFit {
    /// One activation per pattern, then the background's.
    pub h: Vec<f32>,
    /// Σ over all patterns of (W_kc·h_c)² per bin: the modelled power.
    pub total: Vec<f32>,
}

impl Decomposition {
    /// Number of bins of the patterns.
    pub fn bins(&self) -> usize {
        self.patterns.first().map_or(self.background.len(), Vec::len)
    }

    /// All patterns used in a fit: the components, then the background.
    fn columns(&self) -> Vec<&[f32]> {
        let mut cols: Vec<&[f32]> = self.patterns.iter().map(Vec::as_slice).collect();
        if !self.background.is_empty() {
            cols.push(&self.background);
        }
        cols
    }

    /// Fits the loudness of every pattern to one frame of mono `power`.
    pub fn fit_frame(&self, power: &[f32]) -> FrameFit {
        let cols = self.columns();
        let bins = self.bins().min(power.len());
        let v: Vec<f32> = power[..bins].iter().map(|p| p.max(0.0).sqrt()).collect();
        let total_v: f32 = v.iter().sum();
        let mut h = vec![total_v / cols.len().max(1) as f32; cols.len()];
        let mut scratch = vec![0.0f32; bins];
        for _ in 0..INFER_ITERATIONS {
            update_activations(&cols, &v, &mut h, &mut scratch);
        }
        let mut total = vec![0.0f32; bins];
        for (c, w) in cols.iter().enumerate() {
            for k in 0..bins {
                total[k] += (w[k] * h[c]).powi(2);
            }
        }
        FrameFit { h, total }
    }

    /// Expected power of the components `parts` per bin in a fitted frame
    /// (`out`, overwritten): Σ_{c∈parts} (W_kc·h_c)².
    pub fn expected_power(&self, fit: &FrameFit, parts: &[usize], out: &mut [f32]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        for &c in parts {
            let Some(w) = self.patterns.get(c) else { continue };
            for (o, &wk) in out.iter_mut().zip(w) {
                *o += (wk * fit.h[c]).powi(2);
            }
        }
    }

    /// Mask of the components `parts` in a fitted frame (`out`,
    /// overwritten): the amplitude share of the modelled power they explain.
    pub fn mask(&self, fit: &FrameFit, parts: &[usize], out: &mut [f32]) {
        self.expected_power(fit, parts, out);
        for (o, &t) in out.iter_mut().zip(&fit.total) {
            *o = if t > EPS { (*o / t).clamp(0.0, 1.0).sqrt() } else { 0.0 };
        }
    }
}

/// One multiplicative KL update of the activations `h` of a frame with
/// magnitudes `v`, given patterns `cols` that each sum to 1 (so the update's
/// denominator Σ_k W_kc is 1). `model` is scratch space of `v.len()`.
///
/// `h_c ← h_c · Σ_k W_kc · v_k / (W·h)_k`  (Lee & Seung 2001)
fn update_activations(cols: &[&[f32]], v: &[f32], h: &mut [f32], model: &mut [f32]) {
    model.iter_mut().for_each(|m| *m = EPS);
    for (w, &hc) in cols.iter().zip(h.iter()) {
        for (m, &wk) in model.iter_mut().zip(w.iter()) {
            *m += wk * hc;
        }
    }
    for (k, m) in model.iter_mut().enumerate() {
        *m = v[k] / *m;
    }
    for (w, hc) in cols.iter().zip(h.iter_mut()) {
        let num: f32 = w.iter().zip(model.iter()).map(|(a, b)| a * b).sum();
        *hc *= num;
    }
}

/// The result of a search: the patterns and how they are suggested as
/// filters.
#[derive(Clone, Debug)]
pub struct Discovery {
    /// The learned patterns.
    pub decomposition: Decomposition,
    /// Components suggested together, one group per suggested filter,
    /// strongest group first.
    pub groups: Vec<Vec<usize>>,
    /// Share of the learned frames' magnitude explained by each component.
    pub energy: Vec<f32>,
    /// Number of frames the patterns were learned from.
    pub frames_used: usize,
}

/// Frames to learn from, as contiguous blocks `(first frame, count)` inside
/// `first .. end`, together at most about `max_frames` frames.
fn sample_blocks(first: i64, end: i64, max_frames: usize) -> Vec<(i64, usize)> {
    let span = (end - first).max(0) as usize;
    let max_frames = max_frames.max(BLOCK_FRAMES);
    if span <= max_frames {
        return if span == 0 { Vec::new() } else { vec![(first, span)] };
    }
    let blocks = (max_frames / BLOCK_FRAMES).max(1);
    // Evenly spread block starts, the last one ending at `end`.
    let room = (span - BLOCK_FRAMES) as f64;
    (0..blocks)
        .map(|b| {
            let offset = if blocks == 1 {
                0.0
            } else {
                room * b as f64 / (blocks - 1) as f64
            };
            (first + offset.round() as i64, BLOCK_FRAMES)
        })
        .collect()
}

/// A tiny deterministic pseudo-random generator (xorshift), so the same
/// recording always gives the same suggestions.
struct XorShift(u64);

impl XorShift {
    /// Next value in 0..1.
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Runs `f(row index, row)` for every row of `data` (rows of `row_len`)
/// on all cores.
fn par_rows(data: &mut [f32], row_len: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = data.len() / row_len.max(1);
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(rows.max(1));
    let per = rows.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (i, chunk) in data.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (j, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(i * per + j, row);
                }
            });
        }
    });
}

/// Non-negative matrix factorisation of the frames `v` (frame-major
/// magnitudes, `bins` per frame) into `components` learned patterns plus
/// the optional fixed `background` pattern (normalised to sum 1). Returns
/// the patterns (each summing to 1) and the activations (frame-major, one
/// per pattern, the background last).
fn factorize(
    v: &[f32],
    bins: usize,
    components: usize,
    background: Option<&[f32]>,
    iterations: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let frames = v.len() / bins;
    let cols = components + usize::from(background.is_some());
    // Start every pattern at the mean spectrum, perturbed so they can
    // specialise; deterministic seed.
    let mut mean = vec![EPS; bins];
    for row in v.chunks(bins) {
        mean.iter_mut().zip(row).for_each(|(m, x)| *m += x);
    }
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut w: Vec<Vec<f32>> = (0..components)
        .map(|_| mean.iter().map(|m| m * (0.2 + rng.next())).collect())
        .collect();
    if let Some(bg) = background {
        w.push(bg.to_vec());
    }
    for col in &mut w {
        normalise(col);
    }
    let mut h: Vec<f32> = v
        .chunks(bins)
        .flat_map(|row| {
            let s: f32 = row.iter().sum();
            std::iter::repeat_n(s / cols as f32, cols)
        })
        .collect();

    for _ in 0..iterations {
        // Activations, frame by frame (independent given the patterns).
        {
            let refs: Vec<&[f32]> = w.iter().map(Vec::as_slice).collect();
            par_rows(&mut h, cols, |t, ht| {
                let mut scratch = vec![0.0f32; bins];
                update_activations(&refs, &v[t * bins..(t + 1) * bins], ht, &mut scratch);
            });
        }
        // Patterns: W_kc ← W_kc · Σ_t h_tc·v_tk/(WH)_tk / Σ_t h_tc, for the
        // learned columns only. Each thread sums over its frames.
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(frames.max(1));
        let per = frames.div_ceil(threads).max(1);
        let partial: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..frames)
                .step_by(per)
                .map(|t0| {
                    let (w, h) = (&w, &h);
                    s.spawn(move || {
                        let mut num = vec![0.0f32; components * bins];
                        let mut model = vec![0.0f32; bins];
                        for t in t0..(t0 + per).min(frames) {
                            let ht = &h[t * cols..(t + 1) * cols];
                            model.iter_mut().for_each(|m| *m = EPS);
                            for (col, &hc) in w.iter().zip(ht) {
                                model.iter_mut().zip(col).for_each(|(m, wk)| *m += wk * hc);
                            }
                            let vt = &v[t * bins..(t + 1) * bins];
                            for k in 0..bins {
                                model[k] = vt[k] / model[k];
                            }
                            for c in 0..components {
                                let row = &mut num[c * bins..(c + 1) * bins];
                                row.iter_mut().zip(&model).for_each(|(n, r)| *n += ht[c] * r);
                            }
                        }
                        num
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("NMF worker panicked"))
                .collect()
        });
        for c in 0..components {
            let den: f32 = (0..frames).map(|t| h[t * cols + c]).sum::<f32>() + EPS;
            for k in 0..bins {
                let num: f32 = partial.iter().map(|p| p[c * bins + k]).sum();
                w[c][k] *= num / den;
            }
            // Keep each pattern at sum 1 and move its scale into H, so the
            // product is unchanged.
            let s = normalise(&mut w[c]);
            for t in 0..frames {
                h[t * cols + c] *= s;
            }
        }
    }
    w.truncate(components);
    (w, h)
}

/// Scales `v` to sum 1 and returns the old sum (1 if it was zero).
fn normalise(v: &mut [f32]) -> f32 {
    let s: f32 = v.iter().sum();
    if s <= EPS {
        return 1.0;
    }
    v.iter_mut().for_each(|x| *x /= s);
    s
}

/// Learns the sounds of `audio` with the STFT `cfg`. `noise` is the
/// project's noise estimate (its spectrum becomes the fixed background
/// pattern).
pub fn discover(
    audio: &AudioData,
    cfg: StftConfig,
    noise: &NoiseEstimate,
    params: &DiscoverParams,
) -> Result<Discovery> {
    if params.components == 0 {
        bail!("ask for at least one sound");
    }
    let stft = Stft::new(cfg);
    let bins = cfg.bins();
    let total = cfg.frame_count(audio.len()) as i64;
    let (first, end) = match params.range_s {
        Some((a, b)) => {
            let to_frame = |s: f64| ((s.max(0.0) * audio.sample_rate as f64) / cfg.hop as f64).round() as i64;
            (to_frame(a.min(b)).min(total), to_frame(a.max(b)).min(total))
        }
        None => (0, total),
    };
    let blocks = sample_blocks(first, end, params.max_frames);
    if blocks.is_empty() {
        bail!("the recording (or range) is too short to look for sounds");
    }
    let mono = audio.mono();
    let mut power: Vec<Vec<f32>> = Vec::new();
    for &(start, count) in &blocks {
        power.extend(stft.analyze(mono, start, count).power());
    }
    let frames = power.len();
    // The DC bin is left out: a microphone's DC offset or handling rumble
    // there is no sound anyone wants to find, yet it is loud enough to win
    // a pattern of its own.
    let v: Vec<f32> = power
        .iter()
        .flat_map(|f| {
            f.iter()
                .enumerate()
                .map(|(k, p)| if k == 0 { 0.0 } else { p.max(0.0).sqrt() })
        })
        .collect();

    let mut background = match noise {
        NoiseEstimate::Profile(p) if p.len() == bins => p.clone(),
        NoiseEstimate::Adaptive { quantile } => estimate_noise(&power, bins, *quantile),
        NoiseEstimate::Profile(_) => estimate_noise(&power, bins, 0.1),
    };
    background.iter_mut().for_each(|p| *p = p.max(0.0).sqrt());
    if let Some(dc) = background.first_mut() {
        *dc = 0.0;
    }
    let has_background = normalise(&mut background) > EPS * 10.0;
    if !has_background {
        background.clear();
    }

    let (patterns, h) = factorize(
        &v,
        bins,
        params.components,
        has_background.then_some(background.as_slice()),
        params.iterations,
    );
    let cols = params.components + usize::from(has_background);

    // Strongest component first.
    let total_v: f32 = v.iter().sum::<f32>().max(EPS);
    let strength: Vec<f32> = (0..params.components)
        .map(|c| (0..frames).map(|t| h[t * cols + c]).sum::<f32>() / total_v)
        .collect();
    let mut order: Vec<usize> = (0..params.components).collect();
    order.sort_by(|&a, &b| strength[b].total_cmp(&strength[a]));
    let patterns: Vec<Vec<f32>> = order.iter().map(|&c| patterns[c].clone()).collect();
    let energy: Vec<f32> = order.iter().map(|&c| strength[c]).collect();
    let activation = |c: usize| -> Vec<f32> { (0..frames).map(|t| h[t * cols + order[c]]).collect() };

    // Directions from the cells each component explains.
    let directions: Vec<Option<Signature>> = if audio.channels.len() >= 2 {
        let mut sums: Vec<Vec<PairSums>> = (0..params.components)
            .map(|_| (1..audio.channels.len()).map(|_| PairSums::new(bins)).collect())
            .collect();
        let mut share = vec![0.0f32; params.components];
        let mut t0 = 0;
        for &(start, count) in &blocks {
            let spectra: Vec<_> = audio.channels.iter().map(|ch| stft.analyze(ch, start, count)).collect();
            for f in 0..count {
                let t = t0 + f;
                let ht = &h[t * cols..(t + 1) * cols];
                for k in 0..bins {
                    // Each component's share of the modelled power of this
                    // cell is its membership weight.
                    let mut sum = EPS;
                    for c in 0..params.components {
                        share[c] = (patterns[c][k] * ht[order[c]]).powi(2);
                        sum += share[c];
                    }
                    if has_background {
                        sum += (background[k] * ht[cols - 1]).powi(2);
                    }
                    for c in 0..params.components {
                        for (pair, other) in sums[c].iter_mut().zip(&spectra[1..]) {
                            pair.add(k, share[c] / sum, spectra[0].frames[f][k], other.frames[f][k]);
                        }
                    }
                }
            }
            t0 += count;
        }
        sums.iter()
            .map(|s| from_sums(s, 0, cfg.fft_size, params.max_delay_samples))
            .collect()
    } else {
        vec![None; params.components]
    };

    let groups = if params.group {
        let smooth = ((GROUP_SMOOTH_S * audio.sample_rate as f32 / cfg.hop as f32).round() as usize).max(1);
        let curves: Vec<Vec<f32>> = (0..params.components)
            .map(|c| smooth_blocks(&activation(c), &blocks, smooth))
            .collect();
        group(&curves, &directions, params)
    } else {
        (0..params.components).map(|c| vec![c]).collect()
    };

    let labels = patterns
        .iter()
        .map(|p| describe_pattern(p, cfg, audio.sample_rate))
        .collect();
    Ok(Discovery {
        decomposition: Decomposition {
            fft_size: cfg.fft_size,
            patterns,
            background: if has_background { background } else { Vec::new() },
            directions,
            labels,
            level_tolerance_db: params.level_tolerance_db,
            delay_tolerance_samples: params.delay_tolerance_samples,
        },
        groups,
        energy,
        frames_used: frames,
    })
}

/// Moving average of `curve` over ±`radius` frames, within each block of
/// `blocks` (the blocks are not adjacent in time, so averaging across their
/// borders would mix unrelated moments).
fn smooth_blocks(curve: &[f32], blocks: &[(i64, usize)], radius: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(curve.len());
    let mut start = 0;
    for &(_, count) in blocks {
        let block = &curve[start..(start + count).min(curve.len())];
        for i in 0..block.len() {
            let lo = i.saturating_sub(radius);
            let hi = (i + radius + 1).min(block.len());
            out.push(block[lo..hi].iter().sum::<f32>() / (hi - lo) as f32);
        }
        start += count;
    }
    out
}

/// Groups components whose smoothed loudness `curves` correlate and, where
/// `directions` are known, whose directions agree (see the module
/// documentation). Groups are listed in the order of their strongest
/// (lowest-numbered) component.
fn group(curves: &[Vec<f32>], directions: &[Option<Signature>], params: &DiscoverParams) -> Vec<Vec<usize>> {
    let n = curves.len();
    // Union-find over the components.
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        for j in i + 1..n {
            let corr = pearson(&curves[i], &curves[j]);
            let together = match (&directions[i], &directions[j]) {
                (Some(a), Some(b)) => {
                    // Each direction may be off by its own uncertainty.
                    let slack = a
                        .pairs
                        .iter()
                        .zip(&b.pairs)
                        .map(|(x, y)| x.delay_uncertainty.max(y.delay_uncertainty))
                        .fold(0.0f32, f32::max);
                    corr >= params.min_correlation_stereo
                        && agreement(a, b, params.level_tolerance_db, params.delay_tolerance_samples + slack) >= 0.5
                }
                _ => corr >= params.min_correlation_mono,
            };
            if together {
                let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut slot = vec![usize::MAX; n];
    for i in 0..n {
        let r = root(&mut parent, i);
        if slot[r] == usize::MAX {
            slot[r] = groups.len();
            groups.push(Vec::new());
        }
        groups[slot[r]].push(i);
    }
    groups
}

/// Formats a frequency for labels: "440 Hz", "2.6 kHz".
pub fn format_hz(hz: f32) -> String {
    if hz < 1000.0 {
        format!("{:.0} Hz", hz)
    } else {
        format!("{:.1} kHz", hz / 1000.0)
    }
}

/// Describes a magnitude pattern: tonal or noise-like, the band holding the
/// middle 80 % of its power, and its peak, e.g. "tonal, 1.9–3.1 kHz, peak
/// 2.4 kHz".
pub fn describe_pattern(pattern: &[f32], cfg: StftConfig, sample_rate: u32) -> String {
    let power: Vec<f32> = pattern.iter().map(|m| m * m).collect();
    let total: f32 = power.iter().sum();
    if total <= 0.0 {
        return "silent".into();
    }
    let (mut lo, mut hi, mut acc) = (0usize, power.len().saturating_sub(1), 0.0f32);
    let mut found_lo = false;
    for (k, &p) in power.iter().enumerate() {
        acc += p;
        if !found_lo && acc >= 0.1 * total {
            lo = k;
            found_lo = true;
        }
        if acc >= 0.9 * total {
            hi = k;
            break;
        }
    }
    let peak = (0..power.len())
        .max_by(|&a, &b| power[a].total_cmp(&power[b]))
        .unwrap_or(0);
    // Tonal sounds put their power into a few lines: the effective number
    // of bins (Σp)²/Σp² is then small compared with the band they span
    // (padded by the window's main lobe), whereas noise fills its band.
    let sum_sq: f32 = power.iter().map(|p| p * p).sum();
    let effective = total * total / sum_sq.max(1e-30);
    let kind = if effective < 0.3 * (hi.max(lo) - lo + 5) as f32 {
        "tonal"
    } else {
        "noise-like"
    };
    let hz = |k: usize| cfg.bin_to_hz(k as f32, sample_rate);
    format!(
        "{kind}, {}\u{2013}{}, peak {}",
        format_hz(hz(lo)),
        format_hz(hz(hi)),
        format_hz(hz(peak))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::PairCue;

    /// Cosine similarity of two vectors.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-20)
    }

    #[test]
    fn factorisation_recovers_known_patterns() {
        // Two patterns on 8 bins, switched on and off independently.
        let a = [0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let b = [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 1.0, 0.0];
        let mut v = Vec::new();
        for t in 0..60 {
            let (ha, hb) = ((t % 3) as f32, ((t / 4) % 2) as f32 * 2.0);
            v.extend((0..8).map(|k| ha * a[k] + hb * b[k]));
        }
        let (w, _) = factorize(&v, 8, 2, None, 300);
        let best = |p: &[f32]| cosine(&w[0], p).max(cosine(&w[1], p));
        assert!(best(&a) > 0.99, "{w:?}");
        assert!(best(&b) > 0.99, "{w:?}");
    }

    fn two_pattern_decomposition() -> Decomposition {
        Decomposition {
            fft_size: 6,
            patterns: vec![vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.0, 0.5, 0.5]],
            background: vec![0.25; 4],
            directions: vec![None, None],
            labels: vec!["a".into(), "b".into()],
            level_tolerance_db: 3.0,
            delay_tolerance_samples: 2.0,
        }
    }

    #[test]
    fn frame_fit_finds_the_present_pattern_and_masks_are_shares() {
        let d = two_pattern_decomposition();
        // Pattern 0 loud, a little background everywhere.
        let mag = [10.0f32, 10.0, 0.5, 0.5];
        let power: Vec<f32> = mag.iter().map(|m| m * m).collect();
        let fit = d.fit_frame(&power);
        assert!(fit.h[0] > 10.0 * fit.h[1], "{fit:?}");
        let mut m0 = vec![0.0; 4];
        let mut m1 = vec![0.0; 4];
        let mut both = vec![0.0; 4];
        d.mask(&fit, &[0], &mut m0);
        d.mask(&fit, &[1], &mut m1);
        d.mask(&fit, &[0, 1], &mut both);
        assert!(m0[0] > 0.95 && m0[3] < 0.2, "{m0:?}");
        for k in 0..4 {
            // Power shares add up, never beyond 1 (the background has the rest).
            assert!((both[k].powi(2) - m0[k].powi(2) - m1[k].powi(2)).abs() < 1e-5);
            assert!(both[k] <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn silence_gives_zero_masks_not_nan() {
        let d = two_pattern_decomposition();
        let fit = d.fit_frame(&[0.0; 4]);
        let mut m = vec![1.0; 4];
        d.mask(&fit, &[0, 1], &mut m);
        assert!(m.iter().all(|v| *v == 0.0), "{m:?}");
    }

    #[test]
    fn blocks_cover_short_ranges_whole_and_spread_over_long_ones() {
        assert_eq!(sample_blocks(10, 50, 1000), vec![(10, 40)]);
        assert!(sample_blocks(5, 5, 1000).is_empty());
        let b = sample_blocks(0, 100_000, 640);
        assert_eq!(b.len(), 10);
        assert_eq!(b[0], (0, BLOCK_FRAMES));
        assert_eq!(b[9].0 + BLOCK_FRAMES as i64, 100_000);
    }

    #[test]
    fn grouping_needs_timing_and_direction() {
        let rising: Vec<f32> = (0..50).map(|t| (t as f32 * 0.3).sin().abs()).collect();
        let other: Vec<f32> = (0..50).map(|t| (t as f32 * 0.71 + 1.0).cos().abs()).collect();
        let curves = vec![rising.clone(), rising.iter().map(|v| v * 0.5 + 0.01).collect(), other];
        let p = DiscoverParams::default();
        // Mono: the two that rise and fall together are one sound.
        assert_eq!(group(&curves, &[None, None, None], &p), vec![vec![0, 1], vec![2]]);
        // Stereo: from different places they are two sounds.
        let at = |level_db: f32| {
            Some(Signature {
                pairs: vec![PairCue {
                    level_db,
                    delay_samples: 0.0,
                    delay_uncertainty: 0.0,
                }],
            })
        };
        assert_eq!(
            group(&curves, &[at(0.0), at(12.0), at(0.0)], &p),
            vec![vec![0], vec![1], vec![2]]
        );
        assert_eq!(
            group(&curves, &[at(0.0), at(0.5), at(9.0)], &p),
            vec![vec![0, 1], vec![2]]
        );
    }

    #[test]
    fn smoothing_stays_inside_blocks() {
        let s = smooth_blocks(&[0.0, 0.0, 9.0, 9.0], &[(0, 2), (100, 2)], 1);
        assert_eq!(s, vec![0.0, 0.0, 9.0, 9.0]);
    }

    #[test]
    fn labels_tell_tones_from_noise() {
        let cfg = StftConfig { fft_size: 64, hop: 16 };
        let mut tone = vec![0.001f32; 33];
        tone[8] = 1.0;
        let l = describe_pattern(&tone, cfg, 8000);
        assert!(l.starts_with("tonal") && l.contains("peak 1.0 kHz"), "{l}");
        let noise = vec![1.0f32; 33];
        assert!(describe_pattern(&noise, cfg, 8000).starts_with("noise-like"));
    }

    /// A stereo recording: a 600 Hz tone with an overtone at 1200 Hz in
    /// bursts on the left, and a noise band around 3 kHz at other times on
    /// the right, over faint hiss.
    fn scene(sr: u32) -> AudioData {
        let n = sr as usize * 4;
        let mut seed = 12345u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let mut l = vec![0.0f32; n];
        let mut r = vec![0.0f32; n];
        let mut band = 0.0f32;
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let phase = (t * 2.0) % 1.0;
            let tone_on = phase < 0.35;
            let noise_on = (0.5..0.8).contains(&phase);
            let w = 2.0 * std::f32::consts::PI * t;
            let tone = if tone_on {
                0.3 * (600.0 * w).sin() + 0.15 * (1200.0 * w).sin()
            } else {
                0.0
            };
            // A noise band: many sines with scattered phases, 2.8-3.2 kHz.
            band = 0.0;
            if noise_on {
                for j in 0..40 {
                    let f = 2800.0 + 10.0 * j as f32;
                    band += 0.02 * (f * w + j as f32 * 2.399).sin();
                }
            }
            let hiss = 0.003 * rnd();
            l[i] = tone + 0.1 * band + hiss;
            r[i] = 0.1 * tone + band + 0.003 * rnd();
        }
        let _ = band;
        AudioData::new(sr, vec![l, r])
    }

    #[test]
    fn discovers_the_sounds_of_a_simple_stereo_scene() {
        let audio = scene(16000);
        let cfg = StftConfig {
            fft_size: 512,
            hop: 128,
        };
        let params = DiscoverParams {
            components: 3,
            ..Default::default()
        };
        let d = discover(&audio, cfg, &NoiseEstimate::default(), &params).unwrap();
        assert_eq!(d.decomposition.patterns.len(), 3);
        assert_eq!(d.decomposition.directions.len(), 3);
        // Every component has a direction, and the tone and the noise band
        // end up in different suggestions.
        let hz = |k: usize| cfg.bin_to_hz(k as f32, 16000);
        let peak = |c: usize| {
            let p = &d.decomposition.patterns[c];
            hz((0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap())
        };
        let tone = (0..3)
            .find(|&c| (peak(c) - 600.0).abs() < 50.0)
            .expect("a 600 Hz pattern");
        let band = (0..3)
            .find(|&c| (2700.0..3300.0).contains(&peak(c)))
            .expect("a 3 kHz pattern");
        let group_of = |c: usize| d.groups.iter().position(|g| g.contains(&c)).unwrap();
        assert_ne!(group_of(tone), group_of(band), "{:?}", d.groups);
        let (dt, db) = (
            d.decomposition.directions[tone].as_ref().unwrap(),
            d.decomposition.directions[band].as_ref().unwrap(),
        );
        assert!(dt.pairs[0].level_db < -10.0, "tone is left: {dt:?}");
        assert!(db.pairs[0].level_db > 10.0, "noise band is right: {db:?}");
        // Deterministic.
        let again = discover(&audio, cfg, &NoiseEstimate::default(), &params).unwrap();
        assert_eq!(again.decomposition, d.decomposition);
    }

    #[test]
    fn empty_or_zero_requests_are_errors() {
        let audio = AudioData::new(8000, vec![vec![0.0; 10]]);
        let cfg = StftConfig { fft_size: 64, hop: 16 };
        let zero = DiscoverParams {
            components: 0,
            ..Default::default()
        };
        assert!(discover(&audio, cfg, &NoiseEstimate::default(), &zero).is_err());
        let out_of_range = DiscoverParams {
            range_s: Some((5.0, 9.0)),
            ..Default::default()
        };
        assert!(discover(&audio, cfg, &NoiseEstimate::default(), &out_of_range).is_err());
    }
}
