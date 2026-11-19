//! Peak normalization of the processed output.
//!
//! Extracting a quiet sound (a distant bird, a cricket) leaves a file
//! whose loudest sample may be 30 dB below full scale. Normalizing scales
//! the whole output by **one** gain so that its loudest sample reaches a
//! chosen level. A single gain keeps every ratio inside the recording
//! (attack, decay, the balance of partials) exactly as the filters left
//! them; it is not a compressor.
//!
//! The gain is only known once everything is processed, but processing a
//! long recording twice would double the export time. So
//! [`process_all_normalized`] processes once, keeps the unscaled samples
//! in a scratch file next to the output (raw `f32`, so nothing above full
//! scale is clipped before it is scaled down) and then replays them with
//! the gain applied.

use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::engine::process_all;
use crate::{AudioData, Project};

/// Default target of peak normalization in dBFS: one dB of headroom, so
/// that a later conversion to a lossy format or to integer samples does
/// not clip on inter-sample peaks.
pub const DEFAULT_PEAK_DBFS: f32 = -1.0;

/// Lowest and highest target the user interfaces offer, in dBFS.
pub const PEAK_DBFS_RANGE: (f32, f32) = (-40.0, 0.0);

/// Frames (samples per channel) handed to the sink per block while the
/// scratch file is replayed.
const REPLAY_FRAMES: usize = 65_536;

/// Share of the progress the processing pass accounts for; the replay is
/// pure I/O and much faster.
const PROCESS_SHARE: f64 = 0.9;

/// Largest absolute sample value in `channels`. NaN samples are ignored.
pub fn peak<C: AsRef<[f32]>>(channels: &[C]) -> f32 {
    channels
        .iter()
        .flat_map(|c| c.as_ref().iter())
        .fold(0.0_f32, |m, &x| if x.abs() > m { x.abs() } else { m })
}

/// Gain that scales a signal whose loudest sample is `peak` so that it
/// becomes `target_dbfs`. Silence (or a non-finite peak) gets gain 1:
/// there is nothing to scale, and an infinite gain would turn rounding
/// noise into full-scale noise.
pub fn gain_for_peak(peak: f32, target_dbfs: f32) -> f32 {
    if !(peak.is_finite() && peak > 0.0) {
        return 1.0;
    }
    10f32.powf(target_dbfs / 20.0) / peak
}

/// What [`process_all_normalized`] did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalized {
    /// Loudest sample of the processed output before scaling.
    pub peak: f32,
    /// Gain applied (1 when normalization was off or the output silent).
    pub gain: f32,
}

impl Normalized {
    /// The applied gain in dB.
    pub fn gain_db(&self) -> f32 {
        20.0 * self.gain.log10()
    }
}

/// Processes the whole recording like [`process_all`] and hands the output
/// to `sink` in order, as blocks of channel-major samples.
///
/// With `target_dbfs` set, the output is scaled so its loudest sample
/// reaches that level; this needs a scratch file of 4 bytes per sample
/// and channel in `scratch_dir` (pass the output's directory: it has room
/// for the result anyway, unlike `/tmp`, which may live in memory). The
/// file is deleted afterwards, also on error.
///
/// `progress` receives the fraction done, from 0 to 1.
#[allow(clippy::too_many_arguments)]
pub fn process_all_normalized(
    audio: &AudioData,
    project: &Project,
    chunk_frames: usize,
    threads: usize,
    target_dbfs: Option<f32>,
    scratch_dir: &Path,
    mut progress: impl FnMut(f64),
    mut sink: impl FnMut(&[Vec<f32>]) -> Result<()>,
) -> Result<Normalized> {
    let total = audio.len().max(1) as f64;
    let Some(target) = target_dbfs else {
        let mut peak_seen = 0.0_f32;
        process_all(audio, project, chunk_frames, threads, |r| {
            peak_seen = peak_seen.max(peak(&r.audio));
            progress((r.start_sample + r.audio[0].len()) as f64 / total);
            sink(&r.audio)
        })?;
        return Ok(Normalized {
            peak: peak_seen,
            gain: 1.0,
        });
    };

    let channels = audio.channels.len();
    let scratch = tempfile::Builder::new()
        .prefix(".noiseextractor-")
        .suffix(".f32")
        .tempfile_in(scratch_dir)
        .with_context(|| format!("cannot create a scratch file in {}", scratch_dir.display()))?;
    let mut file = BufWriter::new(scratch.as_file());
    let mut peak_seen = 0.0_f32;
    process_all(audio, project, chunk_frames, threads, |r| {
        peak_seen = peak_seen.max(peak(&r.audio));
        // Interleaved, native byte order: the file never leaves this run.
        for i in 0..r.audio.first().map_or(0, Vec::len) {
            for c in &r.audio {
                file.write_all(&c[i].to_ne_bytes())?;
            }
        }
        progress(PROCESS_SHARE * (r.start_sample + r.audio[0].len()) as f64 / total);
        Ok(())
    })?;
    file.flush()?;
    drop(file);

    let gain = gain_for_peak(peak_seen, target);
    let mut reader = BufReader::new(scratch.as_file());
    reader.seek(SeekFrom::Start(0))?;
    let frames = audio.len();
    let mut bytes = vec![0u8; REPLAY_FRAMES * channels * 4];
    let mut done = 0;
    while done < frames {
        let n = REPLAY_FRAMES.min(frames - done);
        let buf = &mut bytes[..n * channels * 4];
        reader.read_exact(buf).context("scratch file ended early")?;
        let mut block = vec![Vec::with_capacity(n); channels];
        for (k, sample) in buf.chunks_exact(4).enumerate() {
            let x = f32::from_ne_bytes([sample[0], sample[1], sample[2], sample[3]]);
            block[k % channels].push(x * gain);
        }
        sink(&block)?;
        done += n;
        progress(PROCESS_SHARE + (1.0 - PROCESS_SHARE) * done as f64 / frames as f64);
    }
    Ok(Normalized { peak: peak_seen, gain })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stereo test signal: two quiet tones of different level.
    fn quiet_stereo(len: usize) -> AudioData {
        let tone = |amp: f32, f: f32| -> Vec<f32> {
            (0..len)
                .map(|i| amp * (i as f32 * f * std::f32::consts::TAU / 16_000.0).sin())
                .collect()
        };
        AudioData::new(16_000, vec![tone(0.05, 440.0), tone(0.02, 660.0)])
    }

    /// Runs the export and collects what the sink got, channel by channel.
    fn run(audio: &AudioData, target: Option<f32>) -> (Vec<Vec<f32>>, Normalized, Vec<f64>) {
        let dir = tempfile::tempdir().unwrap();
        let mut out = vec![Vec::new(); audio.channels.len()];
        let mut steps = Vec::new();
        let n = process_all_normalized(
            audio,
            &Project::default(),
            64,
            2,
            target,
            dir.path(),
            |p| steps.push(p),
            |block| {
                for (o, b) in out.iter_mut().zip(block) {
                    o.extend_from_slice(b);
                }
                Ok(())
            },
        )
        .unwrap();
        // The scratch file is gone.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
        (out, n, steps)
    }

    #[test]
    fn gain_reaches_the_target_and_leaves_silence_alone() {
        let g = gain_for_peak(0.1, -6.0);
        assert!((20.0 * (0.1 * g).log10() + 6.0).abs() < 1e-4);
        assert_eq!(gain_for_peak(0.0, -1.0), 1.0);
        assert_eq!(gain_for_peak(f32::NAN, -1.0), 1.0);
        assert_eq!(peak(&[vec![0.1, -0.7], vec![0.3, f32::NAN]]), 0.7);
    }

    #[test]
    fn normalized_output_is_the_plain_output_times_one_gain() {
        let audio = quiet_stereo(20_000);
        let (plain, off, _) = run(&audio, None);
        assert_eq!(off.gain, 1.0);
        let (scaled, on, steps) = run(&audio, Some(-1.0));
        assert_eq!(on.peak, off.peak);
        assert_eq!(plain[0].len(), audio.len());
        assert_eq!(scaled[0].len(), audio.len());
        // Every sample, in every channel, scaled by the same gain: the
        // ratio between the channels (and within each) is untouched.
        for (p, s) in plain.iter().zip(&scaled) {
            for (&a, &b) in p.iter().zip(s) {
                assert!((a * on.gain - b).abs() <= 1e-6 * on.gain.max(1.0), "{a} {b}");
            }
        }
        assert!((20.0 * peak(&scaled).log10() + 1.0).abs() < 1e-3, "{}", peak(&scaled));
        assert!(on.gain_db() > 20.0, "quiet input is raised: {}", on.gain_db());
        // Progress only grows and ends at 1.
        assert!(steps.windows(2).all(|w| w[0] <= w[1]));
        assert!((steps.last().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sink_errors_stop_the_export_and_the_scratch_file_still_goes() {
        let audio = quiet_stereo(4_000);
        let dir = tempfile::tempdir().unwrap();
        let r = process_all_normalized(
            &audio,
            &Project::default(),
            64,
            1,
            Some(-1.0),
            dir.path(),
            |_| {},
            |_| anyhow::bail!("disk full"),
        );
        assert!(r.is_err());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_missing_scratch_directory_is_an_error_not_a_panic() {
        let audio = quiet_stereo(1_000);
        let r = process_all_normalized(
            &audio,
            &Project::default(),
            64,
            1,
            Some(-1.0),
            Path::new("/nonexistent/noiseextractor"),
            |_| {},
            |_| Ok(()),
        );
        assert!(r.is_err());
    }
}
