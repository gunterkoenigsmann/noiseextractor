//! Background processing of a recording in chunks, nearest to the user's
//! focus first.
//!
//! A one-hour recording does not have to be processed completely before the
//! user can listen or look at it. The timeline is cut into chunks of
//! [`EngineConfig::chunk_frames`] STFT frames. Worker threads repeatedly pick
//! the most urgent chunk that is missing or out of date, where urgency is
//! the distance to the *focus*: the play head (looking ahead more than
//! behind), the current selection and the visible part of the spectrogram.
//!
//! Every change to the filter list bumps a *generation*. Results of older
//! generations stay visible (and audible) until their replacement arrives,
//! so the display never goes blank while the user tweaks a filter.
//!
//! Chunks are independent: each one is analysed with enough padding that
//! overlap-add, the recursive noise estimator and the adaptive noise floor
//! behave as if the whole file had been processed at once (see
//! [`process_chunk`]). Memory stays bounded by an LRU-like cache that keeps
//! the chunks closest to the focus.

use crate::filter::{compute_gains, FilterKind, Project};
use crate::similar::{find_in_range, Occurrence};
use crate::stft::Stft;
use crate::AudioData;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Tuning of the background engine.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// STFT frames per chunk.
    pub chunk_frames: usize,
    /// Number of worker threads.
    pub workers: usize,
    /// Memory (bytes) the cached chunk results may take. How many chunks
    /// that is depends on the STFT and the number of channels, see
    /// [`ChunkLayout::result_bytes`]; a few chunks per worker are always
    /// allowed, or the workers would evict each other's results.
    pub cache_bytes: usize,
}

impl Default for EngineConfig {
    /// 256 frames per chunk (≈ 2.7 s at 48 kHz with the default STFT), one
    /// worker per core except one for the GUI, and 320 MB of cache (about
    /// 150 chunks or 7 minutes of a stereo recording with the default STFT).
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
        Self {
            chunk_frames: 256,
            workers: cores.saturating_sub(1).max(1),
            cache_bytes: 320 << 20,
        }
    }
}

impl EngineConfig {
    /// How many chunk results of `layout` with `bins` bins and `channels`
    /// channels fit into [`EngineConfig::cache_bytes`], but at least four
    /// per worker.
    pub fn max_cached_chunks(&self, layout: ChunkLayout, bins: usize, channels: usize) -> usize {
        let per_chunk = layout.result_bytes(bins, channels).max(1);
        (self.cache_bytes / per_chunk).max(4 * self.workers.max(1))
    }
}

/// Frames before a chunk whose gains are computed and thrown away, so the
/// decision-directed estimator has settled when the chunk starts.
pub const WARMUP_FRAMES: usize = 32;

/// Display levels are stored as `u8` in half-dB steps from this level up.
pub const DB_FLOOR: f32 = -120.0;

/// Converts a level in dBFS to the stored `u8` representation.
pub fn db_to_u8(db: f32) -> u8 {
    ((db - DB_FLOOR) * 2.0).round().clamp(0.0, 255.0) as u8
}

/// Converts a stored `u8` level back to dBFS.
pub fn u8_to_db(v: u8) -> f32 {
    DB_FLOOR + v as f32 * 0.5
}

/// Everything computed for one chunk.
#[derive(Debug)]
pub struct ChunkResult {
    /// Chunk index.
    pub index: usize,
    /// Project generation this result was computed for.
    pub generation: u64,
    /// Global index of the first frame.
    pub first_frame: i64,
    /// Number of frames.
    pub frames: usize,
    /// Bins per frame.
    pub bins: usize,
    /// First output sample.
    pub start_sample: usize,
    /// Processed audio, one vector per channel.
    pub audio: Vec<Vec<f32>>,
    /// Level of the unprocessed mono mix, frame-major (`frame * bins + k`).
    pub original_db: Vec<u8>,
    /// Level after all enabled filters.
    pub output_db: Vec<u8>,
    /// Green overlay mask (0..255).
    pub selected: Vec<u8>,
    /// Red overlay mask (0..255).
    pub inferred: Vec<u8>,
    /// Amplitude of what the analysis found of the picked filter
    /// ([`Project::found_target`](crate::Project::found_target)) in each
    /// frame, on the scale of the samples (see [`power_to_amplitude`]),
    /// for the overview's orange curve. Empty when nothing is picked.
    pub found: Vec<f32>,
}

/// Turns the power `energy` of one frame, summed over the bins
/// `0..=fft_size/2` of our unscaled, Hann-windowed STFT, into the peak
/// amplitude of a sine with that energy: a 0.5-amplitude sine gives 0.5,
/// so the result can be drawn on the same scale as the waveform.
///
/// Parseval: Σ over all N bins |X|² = N · Σ (x·w)², and the one-sided half
/// holds half of it. With Σ w² = 3N/8 for a Hann window, the windowed
/// mean square is 16·E / (3N²); a sine's peak is √2 times its RMS.
pub fn power_to_amplitude(energy: f32, fft_size: usize) -> f32 {
    let n = fft_size as f32;
    (32.0 * energy.max(0.0) / (3.0 * n * n)).sqrt()
}

/// Static description of how a recording is cut into chunks.
#[derive(Clone, Copy, Debug)]
pub struct ChunkLayout {
    /// STFT hop in samples.
    pub hop: usize,
    /// Frames per chunk.
    pub chunk_frames: usize,
    /// Samples in the recording.
    pub total_samples: usize,
}

impl ChunkLayout {
    /// Samples per chunk.
    pub fn chunk_samples(&self) -> usize {
        self.hop * self.chunk_frames
    }

    /// Number of chunks needed to cover the recording.
    pub fn chunk_count(&self) -> usize {
        self.total_samples.div_ceil(self.chunk_samples()).max(1)
    }

    /// Sample range `[start, end)` produced by chunk `c`.
    pub fn sample_range(&self, c: usize) -> (usize, usize) {
        let start = c * self.chunk_samples();
        (
            start.min(self.total_samples),
            (start + self.chunk_samples()).min(self.total_samples),
        )
    }

    /// Approximate memory taken by one [`ChunkResult`] with `bins` bins
    /// and `channels` channels: four one-byte display layers per frame and
    /// bin plus the processed audio as `f32`.
    pub fn result_bytes(&self, bins: usize, channels: usize) -> usize {
        self.chunk_frames * bins * 4 + self.chunk_samples() * channels * 4
    }

    /// Chunk containing `sample`.
    pub fn chunk_of_sample(&self, sample: usize) -> usize {
        (sample / self.chunk_samples()).min(self.chunk_count() - 1)
    }
}

/// Processes chunk `index` of `audio` with `project` (whose event filters
/// must already be analysed, see [`Project::prepare`]).
///
/// Frame ranges used (C = chunk frames, P = overlap padding, W = warm-up,
/// X = the noise context, from [`NOISE_SPAN`](crate::noise::NOISE_SPAN)
/// frames before the grid point at or before cC-P-W to as far after the
/// grid point after (c+1)C+P, see [`crate::noise::NoiseContext::needed`]):
///
/// ```text
///   context  [cC-P-W-X ........................................ (c+1)C+P+X)
///   gains           [cC-P-W ............................ (c+1)C+P)
///   applied              [cC-P ......................... (c+1)C+P)
///   displayed                 [cC ............. (c+1)C)
/// ```
pub fn process_chunk(
    audio: &AudioData,
    mono: &[f32],
    stft: &Stft,
    project: &Project,
    layout: ChunkLayout,
    index: usize,
    generation: u64,
) -> ChunkResult {
    let cfg = stft.config();
    let bins = cfg.bins();
    let c = layout.chunk_frames as i64;
    let first = index as i64 * c;
    let pad = cfg.overlap_frames() as i64;
    let warm = WARMUP_FRAMES as i64;
    // The mono spectrum covers the frames the gains need (with warm-up)
    // and the context the noise estimate of those frames looks at, which
    // depends only on the frames' global positions (see
    // `noise::NoiseContext::needed`).
    let gains_first = first - pad - warm;
    let gains_len = (c + 2 * pad + warm) as usize;
    let needed = crate::noise::NoiseContext::needed(gains_first, gains_len);
    let ctx_start = needed.start.min(gains_first);
    let ctx_len = (needed.end.max(gains_first + gains_len as i64) - ctx_start) as usize;
    let ctx = (gains_first - ctx_start) as usize; // offset of the gains' frames

    let mono_spec = stft.analyze(mono, ctx_start, ctx_len);
    let power = mono_spec.power();
    // Restrict the noise context to frames that really are inside the
    // recording; zeros beyond the ends would drag the estimate down.
    let total_frames = cfg.frame_count(mono.len()) as i64;
    let valid_lo = (-ctx_start).clamp(0, ctx_len as i64) as usize;
    let valid_hi = (total_frames - ctx_start).clamp(valid_lo as i64, ctx_len as i64) as usize;
    let context = crate::noise::NoiseContext::new(&power[valid_lo..valid_hi], ctx_start + valid_lo as i64);

    let g_start = ctx; // offset of frame cC-P-W in `power`
    let g_len = gains_len;
    let occurrences = find_occurrences(project, mono, &audio.channels, stft, (first - pad)..(first + c + pad));
    let applied_first = first - pad;
    let applied_len = (c + 2 * pad) as usize;
    // Multi-channel: the channels are transformed anyway to apply the
    // gains; the same spectra tell the gains where sounds come from.
    let channel_spectra: Vec<crate::stft::Spectra> = if audio.channels.len() > 1 {
        audio
            .channels
            .iter()
            .map(|ch| stft.analyze(ch, applied_first, applied_len))
            .collect()
    } else {
        Vec::new()
    };
    let gains = compute_gains(
        project,
        &power[g_start..g_start + g_len],
        WARMUP_FRAMES,
        &context,
        first - pad,
        &occurrences,
        &channel_spectra,
    );

    let (start_sample, end_sample) = layout.sample_range(index);
    let out_len = end_sample - start_sample;
    let apply = |mut spectra: crate::stft::Spectra| {
        for (frame, g) in spectra.frames.iter_mut().zip(&gains.gain) {
            for (x, &g) in frame.iter_mut().zip(g) {
                *x *= g;
            }
        }
        stft.synthesize(&spectra, start_sample as i64, out_len)
    };
    let audio_out: Vec<Vec<f32>> = if audio.channels.len() == 1 {
        // Mono: the mix *is* the channel, reuse its spectra.
        let off = warm as usize + ctx; // frame cC-P
        let spectra = crate::stft::Spectra {
            first_frame: applied_first,
            frames: mono_spec.frames[off..off + applied_len].to_vec(),
        };
        vec![apply(spectra)]
    } else {
        channel_spectra.into_iter().map(apply).collect()
    };

    // Display layers for frames cC .. (c+1)C.
    let frames = layout.chunk_frames;
    let full_scale = (cfg.fft_size as f32 / 4.0).powi(2); // |X|² of a 0 dBFS sine
    let mut original_db = Vec::with_capacity(frames * bins);
    let mut output_db = Vec::with_capacity(frames * bins);
    let mut selected = Vec::with_capacity(frames * bins);
    let mut inferred = Vec::with_capacity(frames * bins);
    let p_off = (pad + warm) as usize + ctx; // frame cC
    let found = if gains.found.is_empty() {
        Vec::new()
    } else {
        let gi = pad as usize;
        gains.found[gi..gi + frames]
            .iter()
            .map(|&e| power_to_amplitude(e, cfg.fft_size))
            .collect()
    };
    for t in 0..frames {
        let p = &power[p_off + t];
        let gi = pad as usize + t;
        for k in 0..bins {
            let db = 10.0 * (p[k] / full_scale + 1e-14).log10();
            let g = gains.gain[gi][k];
            original_db.push(db_to_u8(db));
            output_db.push(db_to_u8(db + 20.0 * (g + 1e-7).log10()));
            selected.push((gains.selected[gi][k] * 255.0).round() as u8);
            inferred.push((gains.inferred[gi][k] * 255.0).round() as u8);
        }
    }

    ChunkResult {
        index,
        generation,
        first_frame: first,
        frames,
        bins,
        start_sample,
        audio: audio_out,
        original_db,
        output_db,
        selected,
        inferred,
        found,
    }
}

/// Searches, for every event filter (and the preview, last) whose "same
/// sound elsewhere" part matters (enabled filter or drawn overlay, see
/// [`Project::shows`]), the
/// other occurrences overlapping the global frame range `range`.
/// `channels` are the individual recording channels, used to compare where
/// a candidate comes from with where the marked sound came from (ignored
/// for mono recordings).
pub fn find_occurrences(
    project: &Project,
    mono: &[f32],
    channels: &[impl AsRef<[f32]>],
    stft: &Stft,
    range: std::ops::Range<i64>,
) -> Vec<Vec<Occurrence>> {
    project
        .filters
        .iter()
        .chain(project.preview.as_ref())
        .enumerate()
        .map(|(i, f)| match &f.kind {
            FilterKind::Event(e) if e.use_similar && (f.enabled || project.shows(i)) => e
                .analysis
                .as_ref()
                .and_then(|a| a.template.as_ref())
                .map(|tpl| find_in_range(tpl, mono, channels, stft, range.clone(), &e.similar))
                .unwrap_or_default(),
            _ => Vec::new(),
        })
        .collect()
}

/// Processes the whole recording with `threads` threads and hands the
/// chunks to `sink` in order. Used by the command-line tool and for export;
/// memory use is bounded by a few chunks per thread.
pub fn process_all(
    audio: &AudioData,
    project: &Project,
    chunk_frames: usize,
    threads: usize,
    mut sink: impl FnMut(&ChunkResult) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let mono = audio.mono();
    let layout = ChunkLayout {
        hop: project.stft.hop,
        chunk_frames,
        total_samples: audio.len(),
    };
    let threads = threads.max(1);
    let count = layout.chunk_count();
    let mut next = 0;
    while next < count {
        let batch: Vec<usize> = (next..(next + threads).min(count)).collect();
        let results: Vec<ChunkResult> = std::thread::scope(|s| {
            let handles: Vec<_> = batch
                .iter()
                .map(|&i| s.spawn(move || process_chunk(audio, mono, &Stft::new(project.stft), project, layout, i, 0)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("worker panicked"))
                .collect()
        });
        for r in &results {
            sink(r)?;
        }
        next += batch.len();
    }
    Ok(())
}

/// Where the user is looking and listening. All positions in samples.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Focus {
    /// Current play head position.
    pub playhead: Option<usize>,
    /// Selected time range.
    pub selection: Option<(usize, usize)>,
    /// Visible time range of the spectrogram.
    pub view: Option<(usize, usize)>,
}

/// Behind the play head a chunk is this many times less urgent than the same
/// distance ahead of it: we are about to hear what lies ahead.
const BEHIND_PENALTY: f64 = 4.0;

impl Focus {
    /// Urgency of the sample range `[a, b)`: lower is more urgent.
    /// Without any focus the recording is processed front to back.
    pub fn priority(&self, a: usize, b: usize) -> f64 {
        let dist = |lo: usize, hi: usize| -> f64 {
            if b <= lo {
                (lo - b) as f64
            } else if a >= hi {
                (a - hi) as f64
            } else {
                0.0
            }
        };
        let mut best = f64::INFINITY;
        if let Some(p) = self.playhead {
            let d = if b <= p {
                (p - b) as f64 * BEHIND_PENALTY
            } else if a > p {
                (a - p) as f64
            } else {
                0.0
            };
            best = best.min(d);
        }
        if let Some((lo, hi)) = self.selection {
            best = best.min(dist(lo, hi));
        }
        if let Some((lo, hi)) = self.view {
            best = best.min(dist(lo, hi));
        }
        if best.is_infinite() {
            a as f64
        } else {
            best
        }
    }
}

/// Mutable engine state, guarded by one mutex.
struct State {
    /// Project the workers process (prepared, with analyses).
    project: Arc<Project>,
    /// Bumped on every project change; results carry the generation they were computed for.
    generation: u64,
    /// Current focus used for prioritising chunks.
    focus: Focus,
    /// Latest result per chunk (`None`: never computed or evicted).
    slots: Vec<Option<Arc<ChunkResult>>>,
    /// Chunks a worker is computing right now.
    in_progress: HashSet<usize>,
    /// Set when the engine is dropped; workers exit.
    shutdown: bool,
}

/// Data shared between the engine handle and its workers.
struct Shared {
    /// The recording (with the mono mix used for analysis and masks).
    audio: Arc<AudioData>,
    /// Chunk geometry.
    layout: ChunkLayout,
    /// Engine tuning.
    config: EngineConfig,
    /// Chunk results kept at most, from [`EngineConfig::max_cached_chunks`].
    max_cached: usize,
    /// Everything mutable.
    state: Mutex<State>,
    /// Wakes idle workers when there may be new work.
    wake: Condvar,
    /// Incremented whenever a result is stored; lets the GUI poll cheaply
    /// whether it needs to repaint.
    revision: AtomicU64,
}

/// Handle to the background workers. Dropping it stops them.
pub struct Engine {
    /// State shared with the workers.
    shared: Arc<Shared>,
    /// Worker threads, joined on drop.
    workers: Vec<JoinHandle<()>>,
}

impl Engine {
    /// Starts the workers. The project must be prepared
    /// ([`Project::prepare`]) so its event filters carry their analysis.
    pub fn new(audio: Arc<AudioData>, project: Project, config: EngineConfig) -> Self {
        let layout = ChunkLayout {
            hop: project.stft.hop,
            chunk_frames: config.chunk_frames,
            total_samples: audio.len(),
        };
        let max_cached = config.max_cached_chunks(layout, project.stft.bins(), audio.channels.len());
        let shared = Arc::new(Shared {
            audio,
            layout,
            max_cached,
            state: Mutex::new(State {
                project: Arc::new(project),
                generation: 1,
                focus: Focus::default(),
                slots: vec![None; layout.chunk_count()],
                in_progress: HashSet::new(),
                shutdown: false,
            }),
            wake: Condvar::new(),
            revision: AtomicU64::new(0),
            config,
        });
        let workers = (0..shared.config.workers.max(1))
            .map(|i| {
                let s = shared.clone();
                std::thread::Builder::new()
                    .name(format!("nx-worker-{i}"))
                    .spawn(move || worker(s))
                    .expect("cannot spawn worker thread")
            })
            .collect();
        Self { shared, workers }
    }

    /// The recording being processed.
    pub fn audio(&self) -> &Arc<AudioData> {
        &self.shared.audio
    }

    /// The mono mix used for analysis.
    pub fn mono(&self) -> &[f32] {
        self.shared.audio.mono()
    }

    /// How many chunk results the engine keeps at most.
    pub fn max_cached_chunks(&self) -> usize {
        self.shared.max_cached
    }

    /// How the recording is cut into chunks.
    pub fn layout(&self) -> ChunkLayout {
        self.shared.layout
    }

    /// Current project.
    pub fn project(&self) -> Arc<Project> {
        self.lock().project.clone()
    }

    /// Replaces the project (it must be prepared and use the same STFT
    /// configuration). All chunks become out of date but stay visible until
    /// recomputed.
    pub fn set_project(&self, project: Project) {
        let mut st = self.lock();
        assert_eq!(project.stft, st.project.stft, "changing the STFT requires a new engine");
        st.project = Arc::new(project);
        st.generation += 1;
        drop(st);
        self.shared.wake.notify_all();
    }

    /// Updates the focus, which re-prioritises pending work.
    pub fn set_focus(&self, focus: Focus) {
        let mut st = self.lock();
        if st.focus != focus {
            st.focus = focus;
            drop(st);
            self.shared.wake.notify_all();
        }
    }

    /// Latest result for chunk `c`, possibly from an older generation.
    pub fn chunk(&self, c: usize) -> Option<Arc<ChunkResult>> {
        self.lock().slots.get(c).cloned().flatten()
    }

    /// Like [`Engine::chunk`] but never blocks; for the audio callback.
    pub fn try_chunk(&self, c: usize) -> Option<Option<Arc<ChunkResult>>> {
        self.shared
            .state
            .try_lock()
            .ok()
            .map(|st| st.slots.get(c).cloned().flatten())
    }

    /// Counter that changes whenever a new result has been stored.
    pub fn revision(&self) -> u64 {
        self.shared.revision.load(Ordering::Acquire)
    }

    /// (chunks up to date, chunks cached, total chunks).
    pub fn progress(&self) -> (usize, usize, usize) {
        let st = self.lock();
        let cached = st.slots.iter().flatten().count();
        let current = st
            .slots
            .iter()
            .flatten()
            .filter(|r| r.generation == st.generation)
            .count();
        (current, cached, st.slots.len())
    }

    /// Whether chunk `c` holds a result for the current project.
    pub fn is_current(&self, c: usize) -> bool {
        let st = self.lock();
        st.slots
            .get(c)
            .and_then(|s| s.as_ref())
            .is_some_and(|r| r.generation == st.generation)
    }

    /// Locks the state; a poisoned lock (a worker panicked) is still usable
    /// because every update leaves the state consistent.
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.shared.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for Engine {
    /// Stops and joins the workers.
    fn drop(&mut self) {
        self.lock().shutdown = true;
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Chooses the next chunk to compute, evicting a far-away cached chunk if
/// the cache is full. Returns `None` if there is nothing worth doing.
fn pick_job(st: &mut State, layout: ChunkLayout, max_cached: usize) -> Option<usize> {
    let prio = |c: usize| {
        let (a, b) = layout.sample_range(c);
        st.focus.priority(a, b)
    };
    let candidate = (0..st.slots.len())
        .filter(|c| !st.in_progress.contains(c))
        .filter(|&c| st.slots[c].as_ref().is_none_or(|r| r.generation != st.generation))
        .map(|c| (prio(c), c))
        .min_by(|a, b| a.0.total_cmp(&b.0))?;
    if st.slots[candidate.1].is_some() {
        return Some(candidate.1); // refreshing a cached chunk needs no room
    }
    let cached = st.slots.iter().filter(|s| s.is_some()).count() + st.in_progress.len();
    if cached < max_cached {
        return Some(candidate.1);
    }
    // Cache full: only worth it if we can drop something less urgent.
    let worst = (0..st.slots.len())
        .filter(|&c| st.slots[c].is_some() && !st.in_progress.contains(&c))
        .map(|c| (prio(c), c))
        .max_by(|a, b| a.0.total_cmp(&b.0))?;
    if worst.0 > candidate.0 {
        st.slots[worst.1] = None;
        Some(candidate.1)
    } else {
        None
    }
}

/// Worker thread main loop.
fn worker(shared: Arc<Shared>) {
    let mut stft: Option<Stft> = None;
    loop {
        let (job, project, generation) = {
            let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if st.shutdown {
                    return;
                }
                if let Some(c) = pick_job(&mut st, shared.layout, shared.max_cached) {
                    st.in_progress.insert(c);
                    break (c, st.project.clone(), st.generation);
                }
                st = shared.wake.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        };
        let stft = stft.get_or_insert_with(|| Stft::new(project.stft));
        let result = process_chunk(
            &shared.audio,
            shared.audio.mono(),
            stft,
            &project,
            shared.layout,
            job,
            generation,
        );
        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        st.in_progress.remove(&job);
        // A result from an older project only replaces something even older.
        if st.slots[job].as_ref().is_none_or(|r| r.generation <= generation) {
            st.slots[job] = Some(Arc::new(result));
            shared.revision.fetch_add(1, Ordering::AcqRel);
        }
        drop(st);
        // Another worker may be waiting for cache room or for this chunk.
        shared.wake.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Selection;
    use crate::filter::{Filter, FilterAction};
    use crate::stft::StftConfig;
    use std::time::{Duration, Instant};

    fn noisy_tone(sr: u32, secs: f32) -> AudioData {
        let mut rng = 99u64;
        let x = (0..(sr as f32 * secs) as usize)
            .map(|i| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let n = ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05;
                let t = i as f32 / sr as f32;
                // tone bursts every half second
                let burst = if (t * 2.0).fract() < 0.25 { 0.5 } else { 0.0 };
                burst * (2.0 * std::f32::consts::PI * 440.0 * t).sin() + n
            })
            .collect();
        AudioData::new(sr, vec![x])
    }

    fn small_project() -> Project {
        Project {
            stft: StftConfig {
                fft_size: 512,
                hop: 128,
            },
            ..Project::default()
        }
    }

    #[test]
    fn chunked_output_matches_single_chunk_without_filters() {
        let audio = noisy_tone(8000, 3.0);
        let mut p = small_project();
        p.filters.clear();
        let mut out = Vec::new();
        process_all(&audio, &p, 16, 3, |r| {
            out.extend_from_slice(&r.audio[0]);
            Ok(())
        })
        .unwrap();
        assert_eq!(out.len(), audio.len());
        for (a, b) in audio.channels[0].iter().zip(&out) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn chunk_size_does_not_change_noise_reduction_much() {
        // Seams between chunks must be inaudible: compare two chunkings.
        let audio = noisy_tone(8000, 3.0);
        let p = small_project();
        let run = |frames| {
            let mut out = Vec::new();
            process_all(&audio, &p, frames, 2, |r| {
                out.extend_from_slice(&r.audio[0]);
                Ok(())
            })
            .unwrap();
            out
        };
        let (a, b) = (run(32), run(48));
        let err: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32;
        let sig: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32;
        assert!(
            err < sig * 0.01,
            "chunking changed the output by {} dB",
            10.0 * (err / sig).log10()
        );
    }

    #[test]
    fn found_power_is_converted_to_a_sines_peak_amplitude() {
        // Mid-recording frame of a 0.5-amplitude sine, between bins on
        // purpose so the energy is spread over several bins.
        let cfg = StftConfig {
            fft_size: 512,
            hop: 128,
        };
        let x: Vec<f32> = (0..8000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1234.5 * i as f32 / 8000.0).sin())
            .collect();
        let power = Stft::new(cfg).analyze(&x, 30, 1).power();
        let a = power_to_amplitude(power[0].iter().sum(), cfg.fft_size);
        assert!((a - 0.5).abs() < 0.01, "amplitude {a}");
        assert_eq!(power_to_amplitude(0.0, 512), 0.0);
    }

    #[test]
    fn the_picked_noise_reduction_finds_the_noise_not_the_tone() {
        let audio = noisy_tone(8000, 3.0);
        let found = |p: &Project| {
            let mut found = Vec::new();
            process_all(&audio, p, 32, 2, |r| {
                found.extend_from_slice(&r.found);
                Ok(())
            })
            .unwrap();
            found
        };
        let mut p = small_project();
        assert!(found(&p).is_empty(), "nothing picked, nothing measured");
        p.highlight = Some(0);
        let f = found(&p);
        assert_eq!(
            f.len(),
            32 * ChunkLayout {
                hop: 128,
                chunk_frames: 32,
                total_samples: audio.len()
            }
            .chunk_count()
        );
        // Frame i is centred on sample 128·i. Noise only at 0.30–0.45 s;
        // a 0.5 tone burst plus the same noise at 0.02–0.10 s. The noise
        // is uniform ±0.025: RMS 0.0144, i.e. a "sine peak" of 0.02.
        let mean = |r: std::ops::Range<usize>| f[r.clone()].iter().sum::<f32>() / r.len() as f32;
        let quiet = mean(2400 / 128..3600 / 128);
        let burst = mean(160 / 128 + 1..800 / 128);
        assert!((0.008..0.03).contains(&quiet), "noise found between bursts: {quiet}");
        assert!(burst < 0.05, "the tone is not noise: {burst}");
    }

    /// Issue #3: with slowly changing noise, each chunk used to estimate its
    /// own noise spectrum, so the gain stepped at chunk borders and the
    /// output depended on the chunk size. Now a frame's noise estimate
    /// depends only on its position, so two chunkings agree.
    #[test]
    fn slowly_changing_noise_gives_the_same_output_for_any_chunking() {
        let sr = 8000;
        let secs = 12.0;
        let mut rng = 7u64;
        let x: Vec<f32> = (0..(sr as f32 * secs) as usize)
            .map(|i| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let t = i as f32 / sr as f32;
                // The noise grows eightfold over the recording.
                let level = 0.01 * 8f32.powf(t / secs);
                let n = ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * level;
                let burst = if (t * 2.0).fract() < 0.25 { 0.3 } else { 0.0 };
                burst * (2.0 * std::f32::consts::PI * 440.0 * t).sin() + n
            })
            .collect();
        let audio = AudioData::new(sr, vec![x]);
        let p = small_project();
        let run = |frames| {
            let mut out = Vec::new();
            process_all(&audio, &p, frames, 2, |r| {
                out.extend_from_slice(&r.audio[0]);
                Ok(())
            })
            .unwrap();
            out
        };
        let (a, b) = (run(32), run(100));
        let err: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32;
        let sig: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32;
        let db = 10.0 * (err / sig).log10();
        // Per-chunk estimates gave about -44 dB here; now only the
        // estimator's warm-up could differ, and it settles completely.
        assert!(db < -80.0, "chunking changed the output by {db} dB");
    }

    #[test]
    fn noise_reduction_lowers_noise_between_bursts_and_keeps_bursts() {
        let audio = noisy_tone(8000, 3.0);
        let mut out = Vec::new();
        process_all(&audio, &small_project(), 32, 2, |r| {
            out.extend_from_slice(&r.audio[0]);
            Ok(())
        })
        .unwrap();
        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        // 0.30–0.45 s is noise only, 0.02–0.10 s is inside a burst.
        let quiet_in = rms(&audio.channels[0][2400..3600]);
        let quiet_out = rms(&out[2400..3600]);
        assert!(
            quiet_out < quiet_in * 0.3,
            "noise only reduced from {quiet_in} to {quiet_out}"
        );
        let loud_in = rms(&audio.channels[0][4100..4800]);
        let loud_out = rms(&out[4100..4800]);
        assert!((loud_out / loud_in) > 0.9, "burst lost energy: {loud_in} -> {loud_out}");
    }

    /// A gliding call (fundamental 900 → 1500 Hz from 0.2 to 1.3 s, with
    /// 2nd and 3rd harmonics) and a whistle at 2400 Hz from 0.5 to 1.0 s
    /// that the 2nd harmonic crosses around 0.75 s. Returns the call alone and the mix.
    fn call_and_whistle(sr: u32) -> (Vec<f32>, Vec<f32>) {
        let n = (1.5 * sr as f32) as usize;
        let mut phase = 0.0f32;
        let call: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                if !(0.2..1.3).contains(&t) {
                    return 0.0;
                }
                let u = (t - 0.2) / 1.1;
                phase += 2.0 * std::f32::consts::PI * (900.0 + 600.0 * u) / sr as f32;
                let env = (std::f32::consts::PI * u).sin();
                env * (0.4 * phase.sin() + 0.2 * (2.0 * phase).sin() + 0.12 * (3.0 * phase).sin())
            })
            .collect();
        let mut rng = 3u64;
        let mix = call
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let noise = ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.002;
                let t = i as f32 / sr as f32;
                // The whistle sounds from 0.5 to 1.0 s, with soft edges.
                let w = if (0.5..1.0).contains(&t) {
                    (std::f32::consts::PI * (t - 0.5) / 0.5).sin().sqrt()
                } else {
                    0.0
                };
                c + 0.3 * w * (2.0 * std::f32::consts::PI * 2400.0 * t).sin() + noise
            })
            .collect();
        (call, mix)
    }

    /// Extracts the call and suppresses the whistle of [`call_and_whistle`],
    /// in mono or with the call on the left and the whistle on the right.
    /// Returns how far (dB) the overtone at the crossing is from the call
    /// alone (in the call's louder channel, the first), and what fraction of
    /// the whistle's power is left where it sounds alone (in its louder
    /// channel, the last).
    fn crossing(stereo: bool) -> (f32, f32) {
        let sr = 16000;
        let (call, mix) = call_and_whistle(sr);
        let whistle: Vec<f32> = mix.iter().zip(&call).map(|(m, c)| m - c).collect();
        let audio = if stereo {
            let delayed = |x: &[f32], d: usize, g: f32| -> Vec<f32> {
                (0..x.len()).map(|i| if i >= d { g * x[i - d] } else { 0.0 }).collect()
            };
            let (wl, cr) = (delayed(&whistle, 4, 0.5), delayed(&call, 3, 0.5));
            let left = call.iter().zip(&wl).map(|(c, w)| c + w).collect();
            let right = cr.iter().zip(&whistle).map(|(c, w)| c + w).collect();
            AudioData::new(sr, vec![left, right])
        } else {
            AudioData::new(sr, vec![mix])
        };
        let mut p = small_project();
        p.filters = vec![
            Filter::event(
                "call",
                Selection {
                    start_s: 0.2,
                    end_s: 1.3,
                    low_hz: 850.0,
                    high_hz: 1550.0,
                },
                FilterAction::Extract,
            ),
            Filter::event(
                "whistle",
                Selection {
                    start_s: 0.5,
                    end_s: 1.0,
                    low_hz: 2300.0,
                    high_hz: 2500.0,
                },
                FilterAction::Suppress,
            ),
        ];
        p.prepare(&audio);
        let last = audio.channels.len() - 1;
        let (mut out, mut out_last) = (Vec::new(), Vec::new());
        process_all(&audio, &p, 32, 2, |r| {
            out.extend_from_slice(&r.audio[0]);
            out_last.extend_from_slice(&r.audio[last]);
            Ok(())
        })
        .unwrap();
        // Power around 2400 Hz in a time window, from an STFT of `x`.
        let stft = Stft::new(p.stft);
        let band = |x: &[f32], t0: f32, t1: f32| -> f32 {
            let (a, b) = ((t0 * sr as f32 / 128.0) as i64, (t1 * sr as f32 / 128.0) as i64);
            let power = stft.analyze(x, a, (b - a) as usize).power();
            let k = p.stft.hz_to_bin(2400.0, sr).round() as usize;
            power.iter().map(|f| f[k - 3..=k + 3].iter().sum::<f32>()).sum()
        };
        let db = 10.0 * (band(&out, 0.65, 0.85) / band(&call, 0.65, 0.85)).log10();
        let left = band(&out_last, 0.55, 0.65) / band(&audio.channels[last], 0.55, 0.65);
        (db, left)
    }

    #[test]
    fn crossing_whistle_is_removed_but_the_kept_calls_overtone_stays() {
        // Without sharing the bins, the overtone lost ~28 dB at the crossing.
        for stereo in [false, true] {
            let (db, left) = crossing(stereo);
            assert!(
                db.abs() < 4.0,
                "stereo {stereo}: overtone {db:.1} dB off the call alone"
            );
            assert!(left < 0.05, "stereo {stereo}: whistle only reduced to {left:.3}");
        }
    }

    #[test]
    fn stereo_channels_are_processed_separately() {
        let mono = noisy_tone(8000, 1.0);
        let left = mono.channels[0].to_vec();
        let right: Vec<f32> = left.iter().map(|x| -0.5 * x).collect();
        let audio = AudioData::new(8000, vec![left, right]);
        let mut p = small_project();
        p.filters.clear();
        let mut out = [Vec::new(), Vec::new()];
        process_all(&audio, &p, 16, 1, |r| {
            out[0].extend_from_slice(&r.audio[0]);
            out[1].extend_from_slice(&r.audio[1]);
            Ok(())
        })
        .unwrap();
        for i in 0..audio.len() {
            assert!((out[1][i] - audio.channels[1][i]).abs() < 1e-4);
        }
    }

    #[test]
    fn suppress_event_filter_removes_marked_burst_and_its_repetitions() {
        let audio = noisy_tone(8000, 2.0);
        let run = |use_similar: bool| {
            let mut p = small_project();
            p.filters.clear();
            let mut f = Filter::event(
                "tone",
                Selection {
                    start_s: 0.5,
                    end_s: 0.75,
                    low_hz: 380.0,
                    high_hz: 500.0,
                },
                FilterAction::Suppress,
            );
            f.set_part_enabled(3, use_similar);
            p.filters.push(f);
            p.prepare(&audio);
            let mut out = Vec::new();
            process_all(&audio, &p, 32, 2, |r| {
                out.extend_from_slice(&r.audio[0]);
                Ok(())
            })
            .unwrap();
            out
        };
        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let original = &audio.channels[0];
        // Without "same sound elsewhere": the marked burst (0.5–0.75 s) is
        // gone, the next identical one (1.0–1.25 s) is not.
        let only_marked = run(false);
        assert!(rms(&only_marked[4300..5600]) < 0.3 * rms(&original[4300..5600]));
        assert!(rms(&only_marked[8300..9600]) > 0.9 * rms(&original[8300..9600]));
        // With it, every repetition of the burst is removed too.
        let all = run(true);
        for start in [300, 4300, 8300, 12300] {
            let range = start..start + 1300;
            assert!(
                rms(&all[range.clone()]) < 0.3 * rms(&original[range.clone()]),
                "burst at sample {start} survived"
            );
        }
    }

    #[test]
    fn focus_priorities_prefer_ahead_of_playhead() {
        let f = Focus {
            playhead: Some(1000),
            ..Default::default()
        };
        assert_eq!(f.priority(900, 1100), 0.0);
        assert!(f.priority(1100, 1200) < f.priority(800, 900));
        let none = Focus::default();
        assert!(none.priority(0, 10) < none.priority(10, 20));
        let sel = Focus {
            selection: Some((5000, 6000)),
            playhead: Some(0),
            ..Default::default()
        };
        assert_eq!(sel.priority(5500, 5600), 0.0);
    }

    #[test]
    fn pick_job_respects_cache_limit_and_evicts_far_chunks() {
        let layout = ChunkLayout {
            hop: 1,
            chunk_frames: 10,
            total_samples: 100,
        };
        let mut st = State {
            project: Arc::new(small_project()),
            generation: 1,
            focus: Focus {
                playhead: Some(55),
                ..Default::default()
            },
            slots: vec![None; 10],
            in_progress: HashSet::new(),
            shutdown: false,
        };
        assert_eq!(pick_job(&mut st, layout, 2), Some(5));
        st.in_progress.insert(5);
        assert_eq!(pick_job(&mut st, layout, 2), Some(6));
        st.in_progress.clear();
        // Cache holds 0 and 9 (far away); chunk 5 is more urgent -> evict 0.
        let dummy = |i| {
            Some(Arc::new(ChunkResult {
                index: i,
                generation: 1,
                first_frame: 0,
                frames: 0,
                bins: 0,
                start_sample: 0,
                audio: vec![],
                original_db: vec![],
                output_db: vec![],
                selected: vec![],
                inferred: vec![],
                found: vec![],
            }))
        };
        st.slots[0] = dummy(0);
        st.slots[9] = dummy(9);
        assert_eq!(pick_job(&mut st, layout, 2), Some(5));
        assert!(
            st.slots[0].is_none(),
            "the chunk furthest behind the play head goes first"
        );
        // Everything near the focus cached and current: nothing to do.
        st.slots = (0..10)
            .map(|i| if (4..6).contains(&i) { dummy(i) } else { None })
            .collect();
        st.focus = Focus {
            view: Some((40, 60)),
            ..Default::default()
        };
        assert_eq!(pick_job(&mut st, layout, 2), None);
    }

    #[test]
    fn engine_processes_focus_first_and_reacts_to_project_changes() {
        let audio = Arc::new(noisy_tone(8000, 6.0));
        let mut p = small_project();
        p.filters.clear();
        let engine = Engine::new(
            audio.clone(),
            p,
            EngineConfig {
                chunk_frames: 16,
                workers: 2,
                cache_bytes: usize::MAX,
            },
        );
        let total = engine.layout().chunk_count();
        engine.set_focus(Focus {
            playhead: Some(audio.len() - 10),
            ..Default::default()
        });
        let deadline = Instant::now() + Duration::from_secs(20);
        while engine.progress().0 < total {
            assert!(Instant::now() < deadline, "engine did not finish");
            std::thread::sleep(Duration::from_millis(5));
        }
        let rev = engine.revision();
        engine.set_project(small_project());
        while engine.progress().0 < total {
            assert!(Instant::now() < deadline, "engine did not refresh");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(engine.revision() > rev);
        assert!(engine.is_current(0));
        assert!(engine.chunk(total - 1).is_some());
    }

    #[test]
    fn db_u8_conversion_roundtrips_in_half_db_steps() {
        for db in [-120.0, -60.5, -3.0, 0.0] {
            assert!((u8_to_db(db_to_u8(db)) - db).abs() <= 0.25);
        }
        assert_eq!(db_to_u8(-200.0), 0);
    }

    #[test]
    fn cache_budget_counts_bytes_and_keeps_a_few_chunks_per_worker() {
        let layout = ChunkLayout {
            hop: 512,
            chunk_frames: 256,
            total_samples: 1 << 30,
        };
        // 256 frames × 1025 bins × 4 layers + 131072 samples × 2 channels × 4 bytes
        assert_eq!(layout.result_bytes(1025, 2), 1_049_600 + 1_048_576);
        let config = EngineConfig {
            chunk_frames: 256,
            workers: 3,
            cache_bytes: 100 * 2_098_176,
        };
        assert_eq!(config.max_cached_chunks(layout, 1025, 2), 100);
        // A bigger FFT makes chunks bigger, so fewer fit.
        assert_eq!(
            config.max_cached_chunks(layout, 4097, 2),
            100 * 2_098_176 / (256 * 4097 * 4 + 1_048_576)
        );
        let tiny = EngineConfig {
            cache_bytes: 1,
            ..config
        };
        assert_eq!(tiny.max_cached_chunks(layout, 1025, 2), 12);
    }

    #[test]
    fn audio_in_scratch_files_is_processed_like_audio_on_the_heap() {
        let dir = tempfile::tempdir().unwrap();
        let heap = noisy_tone(8000, 2.0);
        let path = dir.path().join("t.wav");
        let right: Vec<f32> = heap.channels[0].iter().map(|x| 0.5 * x).collect();
        let heap = AudioData::new(8000, vec![heap.channels[0].to_vec(), right]);
        heap.save_wav(&path).unwrap();
        let storage = crate::samples::StorageOptions {
            spill_above_bytes: 0,
            dir: dir.path().to_path_buf(),
        };
        let mapped = AudioData::load_with(&path, &storage).unwrap();
        assert!(mapped.is_mapped());
        let run = |audio: &AudioData| {
            let mut out = vec![Vec::new(); 2];
            process_all(audio, &small_project(), 16, 2, |r| {
                for (o, a) in out.iter_mut().zip(&r.audio) {
                    o.extend_from_slice(a);
                }
                Ok(())
            })
            .unwrap();
            out
        };
        assert_eq!(run(&mapped), run(&heap));
    }
}
