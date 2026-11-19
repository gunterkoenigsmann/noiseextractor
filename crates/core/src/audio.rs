//! Loading and saving audio files.
//!
//! Decoding goes through Symphonia (WAV, FLAC, Ogg/Vorbis, MP3, AAC/M4A,
//! ALAC); writing produces 32-bit float WAV via `hound`, so no precision is
//! lost between processing and export.

use crate::samples::{self, Samples, SamplesWriter, StorageOptions};
use anyhow::{bail, Context, Result};
use std::path::Path;

/// An audio file type the program opens, with the names the desktop
/// environments know it by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileType {
    /// Lower-case file name extensions, without the dot.
    pub extensions: &'static [&'static str],
    /// MIME types (freedesktop.org shared-mime-info names) for the Linux
    /// desktop file.
    pub mime_types: &'static [&'static str],
    /// The macOS Uniform Type Identifier for `Info.plist`.
    pub uti: &'static str,
}

/// Every file type [`AudioData::load`] decodes (the Symphonia features
/// enabled in `Cargo.toml`). The packages offer the program under "Open
/// with" for exactly these; `tests/packaging.rs` compares this list with
/// the desktop file, the macOS `Info.plist` and the Windows installer.
pub const FILE_TYPES: &[FileType] = &[
    FileType {
        extensions: &["wav"],
        mime_types: &["audio/x-wav", "audio/wav", "audio/vnd.wave"],
        uti: "com.microsoft.waveform-audio",
    },
    FileType {
        extensions: &["flac"],
        mime_types: &["audio/flac", "audio/x-flac"],
        uti: "org.xiph.flac",
    },
    FileType {
        extensions: &["ogg", "oga"],
        mime_types: &["audio/ogg", "audio/x-vorbis+ogg"],
        uti: "org.xiph.ogg-audio",
    },
    FileType {
        extensions: &["mp3"],
        mime_types: &["audio/mpeg"],
        uti: "public.mp3",
    },
    // MP4 audio: AAC or ALAC inside.
    FileType {
        extensions: &["m4a"],
        mime_types: &["audio/mp4", "audio/x-m4a"],
        uti: "public.mpeg-4-audio",
    },
    // Raw AAC in ADTS frames.
    FileType {
        extensions: &["aac"],
        mime_types: &["audio/aac", "audio/x-aac"],
        uti: "public.aac-audio",
    },
];

/// The patterns of the open dialog's "Audio files" entry: every extension
/// of [`FILE_TYPES`] in lower and upper case (`*.wav;*.WAV;...`), since
/// GTK matches patterns case-sensitively.
pub fn file_dialog_patterns() -> String {
    FILE_TYPES
        .iter()
        .flat_map(|t| t.extensions)
        .flat_map(|e| [format!("*.{e}"), format!("*.{}", e.to_uppercase())])
        .collect::<Vec<_>>()
        .join(";")
}

/// A decoded recording: one sample vector per channel, all equally long,
/// plus their mono mix.
///
/// Long recordings keep their samples in mapped scratch files rather than
/// on the heap (see [`crate::samples`]), so their length is not limited by
/// the RAM. Cloning is cheap: the samples are shared.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioData {
    /// Samples per second.
    pub sample_rate: u32,
    /// Channel-major samples in the range -1..1. Treat them as read-only:
    /// the mono mix is computed once, when the recording is made.
    pub channels: Vec<Samples>,
    /// Average of all channels (the first channel itself for mono files).
    mono: Samples,
}

impl AudioData {
    /// Builds a recording from channel vectors. Shorter channels are padded
    /// with silence so that all channels have the same length.
    pub fn new(sample_rate: u32, mut channels: Vec<Vec<f32>>) -> Self {
        let len = channels.iter().map(Vec::len).max().unwrap_or(0);
        for c in &mut channels {
            c.resize(len, 0.0);
        }
        let channels: Vec<Samples> = channels.into_iter().map(Samples::from).collect();
        // Heap channels are mixed on the heap, which cannot fail.
        let mono = samples::mix(&channels, &StorageOptions::default()).unwrap_or_else(|_| Vec::new().into());
        Self {
            sample_rate,
            channels,
            mono,
        }
    }

    /// Number of samples per channel.
    pub fn len(&self) -> usize {
        self.channels.first().map_or(0, |c| c.len())
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
    /// stereo image survives. Computed once, so this is free.
    pub fn mono(&self) -> &[f32] {
        &self.mono
    }

    /// Whether the samples live in mapped scratch files instead of the heap.
    pub fn is_mapped(&self) -> bool {
        self.channels.iter().any(Samples::is_mapped)
    }

    /// Decodes any audio file Symphonia understands, keeping long
    /// recordings in scratch files ([`StorageOptions::default`]).
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with(path, &StorageOptions::default())
    }

    /// Decodes any audio file Symphonia understands; `storage` tells when
    /// and where the samples go to scratch files instead of the heap.
    pub fn load_with(path: &Path, storage: &StorageOptions) -> Result<Self> {
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
        let track = format.default_track().context("file contains no audio track")?.clone();
        let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
        let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
        // Known length (not every format says): reserve exactly, so the
        // vectors do not grow by doubling and leave capacity unused.
        let expected = track.codec_params.n_frames.unwrap_or(0) as usize;
        let mut channels: Vec<SamplesWriter> = Vec::new();
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
                // Plus one vector for the mono mix of several channels.
                let vectors = nch + usize::from(nch > 1);
                let fits = expected.saturating_mul(vectors * 4) <= storage.spill_above_bytes;
                channels = (0..nch)
                    .map(|_| SamplesWriter::new(if fits { expected } else { 0 }))
                    .collect();
            }
            let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
            buf.copy_interleaved_ref(decoded);
            for frame in buf.samples().chunks(nch) {
                for (c, &s) in channels.iter_mut().zip(frame) {
                    c.push(s)?;
                }
            }
            let vectors = channels.len() + usize::from(channels.len() > 1);
            let bytes = channels[0].len().saturating_mul(vectors * 4);
            if bytes > storage.spill_above_bytes && !channels[0].is_spilled() {
                for c in &mut channels {
                    c.spill(&storage.dir)?;
                }
            }
        }
        if channels.is_empty() || sample_rate == 0 {
            bail!("no audio could be decoded from {}", path.display());
        }
        // A channel cut short by a damaged packet is padded with silence.
        let len = channels.iter().map(SamplesWriter::len).max().unwrap_or(0);
        for c in &mut channels {
            while c.len() < len {
                c.push(0.0)?;
            }
        }
        let channels = channels
            .into_iter()
            .map(SamplesWriter::finish)
            .collect::<Result<Vec<_>>>()?;
        let mono = samples::mix(&channels, storage)?;
        Ok(Self {
            sample_rate,
            channels,
            mono,
        })
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
    /// Underlying writer; the header is fixed up in `finish`.
    writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
}

impl WavStreamWriter {
    /// Creates a 32-bit float WAV file.
    pub fn create(path: &Path, sample_rate: u32, channels: u16) -> Result<Self> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        Ok(Self {
            writer: hound::WavWriter::create(path, spec).with_context(|| format!("cannot write {}", path.display()))?,
        })
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
    fn file_dialog_offers_every_extension_in_both_cases() {
        let patterns = file_dialog_patterns();
        let list: Vec<&str> = patterns.split(';').collect();
        for t in FILE_TYPES {
            for e in t.extensions {
                assert_eq!(*e, e.to_lowercase(), "extensions are listed lower-case");
                assert!(list.contains(&format!("*.{e}").as_str()), "{patterns}");
                assert!(list.contains(&format!("*.{}", e.to_uppercase()).as_str()), "{patterns}");
            }
        }
        assert!(!patterns.contains('|'), "would break the wxWidgets wildcard");
    }

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

    #[test]
    fn long_recordings_are_loaded_into_scratch_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.wav");
        let n = 50_000;
        let left: Vec<f32> = (0..n).map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0).collect();
        let right: Vec<f32> = left.iter().rev().copied().collect();
        AudioData::new(44100, vec![left, right]).save_wav(&path).unwrap();
        let heap = AudioData::load(&path).unwrap();
        assert!(!heap.is_mapped(), "a short file stays on the heap");
        // Spill after a bit more than one channel's worth: the samples
        // decoded before the spill must end up in the file as well.
        let storage = StorageOptions {
            spill_above_bytes: n * 4,
            dir: dir.path().join("scratch"),
        };
        std::fs::create_dir(&storage.dir).unwrap();
        let mapped = AudioData::load_with(&path, &storage).unwrap();
        assert!(mapped.is_mapped());
        assert!(mapped.channels.iter().all(Samples::is_mapped));
        assert_eq!(mapped, heap);
        assert_eq!(mapped.mono(), heap.mono());
        // The scratch files vanish with the recording (on Windows they are
        // visible until closed, on Unix they never are).
        drop(mapped);
        assert_eq!(std::fs::read_dir(&storage.dir).unwrap().count(), 0);
    }

    #[test]
    fn mono_of_a_mono_file_is_its_channel() {
        let a = AudioData::new(8000, vec![vec![0.25, -0.5]]);
        assert_eq!(a.mono().as_ptr(), a.channels[0].as_ptr(), "no copy");
    }
}
