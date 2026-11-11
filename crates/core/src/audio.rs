//! Loading and saving audio files.
//!
//! Decoding goes through Symphonia (WAV, FLAC, Ogg/Vorbis, MP3, AAC/M4A,
//! ALAC); writing produces 32-bit float WAV via `hound`, so no precision is
//! lost between processing and export.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// A decoded recording: one sample vector per channel, all equally long.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioData {
    /// Samples per second.
    pub sample_rate: u32,
    /// Channel-major samples in the range -1..1.
    pub channels: Vec<Vec<f32>>,
}

impl AudioData {
    /// Builds a recording from channel vectors. Shorter channels are padded
    /// with silence so that all channels have the same length.
    pub fn new(sample_rate: u32, mut channels: Vec<Vec<f32>>) -> Self {
        let len = channels.iter().map(Vec::len).max().unwrap_or(0);
        for c in &mut channels {
            c.resize(len, 0.0);
        }
        Self { sample_rate, channels }
    }

    /// Number of samples per channel.
    pub fn len(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    /// True if the recording contains no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        self.len() as f64 / self.sample_rate as f64
    }

    /// Average of all channels; the analysis (and the masks) work on this
    /// mix, and the resulting masks are then applied to every channel so the
    /// stereo image survives.
    pub fn mono(&self) -> Vec<f32> {
        let n = self.channels.len().max(1) as f32;
        let mut out = vec![0.0; self.len()];
        for c in &self.channels {
            for (o, &s) in out.iter_mut().zip(c) {
                *o += s / n;
            }
        }
        out
    }

    /// Decodes any audio file Symphonia understands.
    pub fn load(path: &Path) -> Result<Self> {
        use symphonia::core::audio::SampleBuffer;
        use symphonia::core::codecs::DecoderOptions;
        use symphonia::core::errors::Error as SymError;
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;

        let file = std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
            .with_context(|| format!("unsupported audio format: {}", path.display()))?;
        let mut format = probed.format;
        let track = format
            .default_track()
            .context("file contains no audio track")?
            .clone();
        let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
        let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
        let mut channels: Vec<Vec<f32>> = Vec::new();
        loop {
            let packet = match format.next_packet() {
                Ok(p) => p,
                // End of stream is reported as an I/O error by Symphonia.
                Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(SymError::ResetRequired) => break,
                Err(e) => return Err(e.into()),
            };
            if packet.track_id() != track.id {
                continue;
            }
            let decoded = match decoder.decode(&packet) {
                Ok(d) => d,
                // A corrupt packet should not make the whole file unusable.
                Err(SymError::DecodeError(_)) => continue,
                Err(e) => return Err(e.into()),
            };
            let spec = *decoded.spec();
            sample_rate = spec.rate;
            let nch = spec.channels.count();
            if channels.is_empty() {
                channels = vec![Vec::new(); nch];
            }
            let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
            buf.copy_interleaved_ref(decoded);
            for frame in buf.samples().chunks(nch) {
                for (c, &s) in channels.iter_mut().zip(frame) {
                    c.push(s);
                }
            }
        }
        if channels.is_empty() || sample_rate == 0 {
            bail!("no audio could be decoded from {}", path.display());
        }
        Ok(Self::new(sample_rate, channels))
    }

    /// Writes the recording as a 32-bit float WAV file.
    pub fn save_wav(&self, path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.channels.len() as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(path, spec).with_context(|| format!("cannot write {}", path.display()))?;
        for i in 0..self.len() {
            for c in &self.channels {
                w.write_sample(c[i])?;
            }
        }
        w.finalize()?;
        Ok(())
    }
}

/// Incremental WAV writer used for exporting long recordings chunk by chunk
/// without holding the whole result in memory.
pub struct WavStreamWriter {
    writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
}

impl WavStreamWriter {
    /// Creates a 32-bit float WAV file.
    pub fn create(path: &Path, sample_rate: u32, channels: u16) -> Result<Self> {
        let spec = hound::WavSpec { channels, sample_rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
        Ok(Self { writer: hound::WavWriter::create(path, spec).with_context(|| format!("cannot write {}", path.display()))? })
    }

    /// Appends channel-major samples (all channels equally long).
    pub fn write(&mut self, channels: &[Vec<f32>]) -> Result<()> {
        let len = channels.first().map_or(0, Vec::len);
        for i in 0..len {
            for c in channels {
                self.writer.write_sample(c[i])?;
            }
        }
        Ok(())
    }

    /// Flushes and closes the file, fixing up the header.
    pub fn finish(self) -> Result<()> {
        self.writer.finalize()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_preserves_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wav");
        let a = AudioData::new(22050, vec![vec![0.0, 0.5, -0.5, 0.25], vec![1.0, -1.0, 0.0, 0.125]]);
        a.save_wav(&path).unwrap();
        let b = AudioData::load(&path).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn mono_averages_channels_and_new_pads() {
        let a = AudioData::new(8000, vec![vec![1.0, 1.0], vec![0.0]]);
        assert_eq!(a.len(), 2);
        assert_eq!(a.mono(), vec![0.5, 0.5]);
        assert!((a.duration() - 2.0 / 8000.0).abs() < 1e-12);
    }

    #[test]
    fn loading_garbage_fails_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.wav");
        std::fs::write(&path, b"definitely not audio").unwrap();
        assert!(AudioData::load(&path).is_err());
    }
}
