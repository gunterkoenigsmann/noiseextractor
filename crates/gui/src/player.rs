//! Audio playback through the system's default output device (cpal).
//!
//! The audio callback reads processed samples straight from the engine's
//! chunk cache. It never blocks: if the cache is locked or the chunk at the
//! play head has not been processed yet, it falls back to the original
//! recording for that moment. The engine prioritises the chunks ahead of the
//! play head, so in practice the fallback is only heard right after a seek.
//!
//! A jump while playing (a click into the spectrogram, every grain of a
//! scrub) fades out for a few milliseconds, moves, and fades in again
//! ([`Declick`]), so it does not click.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use noiseextractor_core::{ChunkResult, Engine};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// State shared between the GUI and the audio callback.
pub struct PlayShared {
    /// Whether samples are being produced.
    playing: AtomicBool,
    /// Play head in source samples, as `f64` bits (fractional because the
    /// device rate may differ from the file rate).
    position: AtomicU64,
    /// Sample at which playback stops (`u64::MAX`: end of file).
    stop_at: AtomicU64,
    /// Where to jump to, as `f64` bits, once the fade-out is done
    /// ([`NO_JUMP`]: nowhere). Set by the GUI, taken by the callback.
    jump: AtomicU64,
}

/// Value of [`PlayShared::jump`] when no jump is requested (not the bits
/// of any position the GUI asks for, which are finite and non-negative).
const NO_JUMP: u64 = u64::MAX;

/// Length of the fades around a jump, in output frames (≈ 3 ms at 48 kHz).
const FADE_FRAMES: u32 = 144;

/// Gain envelope around jumps: fades out, tells the caller to jump, fades
/// in. Runs in the audio callback, one call per output frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Declick {
    /// Current gain, 0..1.
    gain: f32,
    /// Gain change per frame.
    step: f32,
    /// Where to jump once the gain reaches 0.
    pending: Option<f64>,
}

impl Declick {
    /// Silent at first (playback starts with a fade-in), fading over
    /// `frames` output frames.
    pub fn new(frames: u32) -> Self {
        Declick {
            gain: 0.0,
            step: 1.0 / frames.max(1) as f32,
            pending: None,
        }
    }

    /// Asks for a jump to `to`; a later request replaces an earlier one
    /// that has not happened yet.
    pub fn request(&mut self, to: f64) {
        self.pending = Some(to);
    }

    /// Playback stopped: the next start fades in.
    pub fn silence(&mut self) {
        self.gain = 0.0;
        self.pending = None;
    }

    /// Gain for the next frame, and the position to jump to before
    /// producing it (once the fade-out is complete).
    pub fn next(&mut self) -> (f32, Option<f64>) {
        if self.pending.is_some() {
            self.gain -= self.step;
            if self.gain <= 0.0 {
                self.gain = 0.0;
                return (0.0, self.pending.take());
            }
            return (self.gain, None);
        }
        self.gain = (self.gain + self.step).min(1.0);
        (self.gain, None)
    }
}

impl PlayShared {
    /// Current play head in source samples.
    pub fn position(&self) -> f64 {
        f64::from_bits(self.position.load(Ordering::Relaxed))
    }

    /// Moves the play head.
    fn set_position(&self, p: f64) {
        self.position.store(p.to_bits(), Ordering::Relaxed);
    }
}

/// An open output stream bound to one engine.
pub struct Player {
    /// Keeps the output stream alive; dropping it stops playback.
    _stream: cpal::Stream,
    /// State shared with the audio callback.
    shared: Arc<PlayShared>,
}

impl Player {
    /// Opens the default output device. Fails when there is none (e.g. on a
    /// headless machine); the GUI then works without playback.
    pub fn new(engine: Arc<Engine>) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no audio output device"))?;
        let supported = device.default_output_config().context("cannot query output format")?;
        let shared = Arc::new(PlayShared {
            playing: AtomicBool::new(false),
            position: AtomicU64::new(0f64.to_bits()),
            stop_at: AtomicU64::new(u64::MAX),
            jump: AtomicU64::new(NO_JUMP),
        });
        let config: cpal::StreamConfig = supported.config();
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => build::<f32>(&device, &config, engine, shared.clone()),
            cpal::SampleFormat::I16 => build::<i16>(&device, &config, engine, shared.clone()),
            cpal::SampleFormat::U16 => build::<u16>(&device, &config, engine, shared.clone()),
            cpal::SampleFormat::I32 => build::<i32>(&device, &config, engine, shared.clone()),
            other => Err(anyhow!("unsupported output sample format {other:?}")),
        }?;
        stream.play().context("cannot start audio stream")?;
        Ok(Self {
            _stream: stream,
            shared,
        })
    }

    /// Starts playing at `from` until `until` (exclusive), both in samples.
    /// While already playing, the move is faded (see [`Declick`]).
    pub fn play(&self, from: usize, until: Option<usize>) {
        self.shared
            .stop_at
            .store(until.map_or(u64::MAX, |u| u as u64), Ordering::Relaxed);
        if self.is_playing() {
            self.shared.jump.store((from as f64).to_bits(), Ordering::Release);
        } else {
            self.shared.set_position(from as f64);
            self.shared.playing.store(true, Ordering::Release);
        }
    }

    /// Stops playback, keeping the play head where it is.
    pub fn pause(&self) {
        self.shared.playing.store(false, Ordering::Release);
    }

    /// Whether playback is running.
    pub fn is_playing(&self) -> bool {
        self.shared.playing.load(Ordering::Acquire)
    }

    /// Current play head in samples.
    pub fn position(&self) -> usize {
        self.shared.position() as usize
    }
}

/// Builds the output stream for device sample type `T`.
fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    engine: Arc<Engine>,
    shared: Arc<PlayShared>,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let out_channels = config.channels as usize;
    let step = engine.audio().sample_rate as f64 / config.sample_rate.0 as f64;
    let mut source = Source {
        audio: engine.audio().clone(),
        engine,
        chunk: None,
    };
    let mut declick = Declick::new(FADE_FRAMES);
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            let playing = shared.playing.load(Ordering::Acquire);
            let mut pos = shared.position();
            let stop = shared.stop_at.load(Ordering::Relaxed).min(source.audio.len() as u64) as f64;
            let jump = shared.jump.swap(NO_JUMP, Ordering::AcqRel);
            if jump != NO_JUMP {
                declick.request(f64::from_bits(jump));
            }
            if !playing {
                declick.silence();
            }
            for frame in data.chunks_mut(out_channels) {
                if !playing || pos >= stop {
                    frame.iter_mut().for_each(|s| *s = T::from_sample(0.0));
                    continue;
                }
                let (gain, to) = declick.next();
                if let Some(to) = to {
                    pos = to;
                }
                for (c, s) in frame.iter_mut().enumerate() {
                    *s = T::from_sample(gain * source.interpolate(c, pos));
                }
                pos += step;
            }
            if playing {
                shared.set_position(pos.min(stop));
                if pos >= stop {
                    shared.playing.store(false, Ordering::Release);
                }
            }
        },
        |e| eprintln!("audio stream error: {e}"),
        None,
    )?;
    Ok(stream)
}

/// Sample source for the audio callback, caching the current chunk.
struct Source {
    /// The recording (fallback when a chunk is missing).
    audio: Arc<noiseextractor_core::AudioData>,
    /// Engine whose processed chunks are played.
    engine: Arc<Engine>,
    /// Chunk used for the previous sample, to avoid a lookup per sample.
    chunk: Option<Arc<ChunkResult>>,
}

impl Source {
    /// Processed sample of output channel `out_ch` at source sample `i`,
    /// falling back to the original recording when no result is available.
    fn sample(&mut self, out_ch: usize, i: usize) -> f32 {
        if i >= self.audio.len() {
            return 0.0;
        }
        // Mono files are played on every output channel; extra output
        // channels repeat the last input channel.
        let ch = out_ch.min(self.audio.channels.len() - 1);
        let c = self.engine.layout().chunk_of_sample(i);
        if self.chunk.as_ref().is_none_or(|r| r.index != c) {
            self.chunk = self.engine.try_chunk(c).flatten();
        }
        match &self.chunk {
            Some(r) if r.index == c => r.audio[ch].get(i - r.start_sample).copied().unwrap_or(0.0),
            _ => self.audio.channels[ch][i],
        }
    }

    /// Linear interpolation between neighbouring samples, used when the
    /// device runs at a different rate than the file.
    fn interpolate(&mut self, out_ch: usize, pos: f64) -> f32 {
        let i = pos.floor() as usize;
        let frac = (pos - i as f64) as f32;
        let a = self.sample(out_ch, i);
        if frac == 0.0 {
            return a;
        }
        a + (self.sample(out_ch, i + 1) - a) * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_with_a_fade_in() {
        let mut d = Declick::new(4);
        let gains: Vec<f32> = (0..6).map(|_| d.next().0).collect();
        assert_eq!(gains, vec![0.25, 0.5, 0.75, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_jump_fades_out_moves_and_fades_in() {
        let mut d = Declick::new(4);
        (0..4).for_each(|_| {
            d.next();
        });
        d.request(100.0);
        d.request(200.0); // the later request wins
        assert_eq!(d.next(), (0.75, None));
        assert_eq!(d.next(), (0.5, None));
        assert_eq!(d.next(), (0.25, None));
        assert_eq!(d.next(), (0.0, Some(200.0)));
        assert_eq!(d.next(), (0.25, None));
        // Stopping forgets a jump and makes the next start fade in.
        d.request(5.0);
        d.silence();
        assert_eq!(d.next(), (0.25, None));
    }
}
