//! Writing the processed recording in the formats people share audio in.
//!
//! Every writer is fed chunk by chunk ([`AudioWriter::write`]) so a
//! one-hour recording never has to be held twice in memory, and every one
//! takes the processor's 32-bit float samples directly:
//!
//! | Format      | Encoder                                  | Notes |
//! |-------------|------------------------------------------|-------|
//! | WAV         | `hound`                                  | 32-bit float, bit exact |
//! | FLAC        | `flacenc` (pure Rust)                    | 24-bit, lossless |
//! | Ogg Vorbis  | `vorbis_rs` (libvorbis/aoTuV, built in)  | quality 6 VBR by default |
//! | MP3         | `mp3lame-encoder` (LAME, built in)       | VBR ≈190 kbit/s by default, mono/stereo |
//! | Opus        | `opus` (libopus, built in) + `ogg`       | 48 kHz (resampled), mono/stereo |
//! | ALAC (.m4a) | `alac-encoder` + our own MP4 writer      | 16-bit, lossless after dithering |
//!
//! All C libraries are compiled from source by their `-sys` crates, so the
//! export works the same on Linux, Windows and macOS without system
//! packages. AAC is deliberately missing: the patent pool still licenses
//! encoders, and the only good free encoder (FDK AAC) has a licence that is
//! incompatible with the GPL. ALAC and every other format here are open and
//! royalty free.

use crate::audio::WavStreamWriter;
use anyhow::{bail, ensure, Context, Result};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// A file format the processed recording can be exported to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    /// 32-bit float WAV: exactly what the processor produced.
    Wav,
    /// FLAC with 24-bit samples.
    Flac,
    /// Vorbis in an Ogg container.
    Vorbis,
    /// MPEG-1/2 Layer III.
    Mp3,
    /// Opus in an Ogg container (`.opus`, RFC 7845).
    Opus,
    /// Apple Lossless in an MP4 container (`.m4a`), 16-bit.
    Alac,
}

impl ExportFormat {
    /// Every format, in the order the save dialog offers them.
    pub const ALL: [ExportFormat; 6] = [
        ExportFormat::Wav,
        ExportFormat::Flac,
        ExportFormat::Vorbis,
        ExportFormat::Mp3,
        ExportFormat::Opus,
        ExportFormat::Alac,
    ];

    /// The file name extension a new file of this format gets (no dot).
    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Wav => "wav",
            ExportFormat::Flac => "flac",
            ExportFormat::Vorbis => "ogg",
            ExportFormat::Mp3 => "mp3",
            ExportFormat::Opus => "opus",
            ExportFormat::Alac => "m4a",
        }
    }

    /// Human-readable name for dialogs and messages.
    pub fn label(self) -> &'static str {
        match self {
            ExportFormat::Wav => "WAV (32-bit float, exact)",
            ExportFormat::Flac => "FLAC (24-bit, lossless)",
            ExportFormat::Vorbis => "Ogg Vorbis",
            ExportFormat::Mp3 => "MP3",
            ExportFormat::Opus => "Opus",
            ExportFormat::Alac => "Apple Lossless (16-bit .m4a)",
        }
    }

    /// The short name used on the command line (`--format`).
    pub fn name(self) -> &'static str {
        match self {
            ExportFormat::Wav => "wav",
            ExportFormat::Flac => "flac",
            ExportFormat::Vorbis => "vorbis",
            ExportFormat::Mp3 => "mp3",
            ExportFormat::Opus => "opus",
            ExportFormat::Alac => "alac",
        }
    }

    /// Parses a [`name`](Self::name) or an extension, ignoring case.
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|f| f.name() == name || f.extension() == name)
            .or(match name.as_str() {
                "oga" => Some(ExportFormat::Vorbis),
                _ => None,
            })
    }

    /// The format a file name asks for by its extension, if any.
    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension().and_then(|e| e.to_str()).and_then(Self::from_name)
    }

    /// True for formats that throw information away; their quality can be
    /// chosen with [`ExportOptions::bitrate_kbps`].
    pub fn is_lossy(self) -> bool {
        matches!(self, ExportFormat::Vorbis | ExportFormat::Mp3 | ExportFormat::Opus)
    }

    /// The largest number of channels the format (or our use of it) takes.
    pub fn max_channels(self) -> usize {
        match self {
            ExportFormat::Wav => usize::from(u16::MAX),
            ExportFormat::Vorbis => 255,
            ExportFormat::Flac | ExportFormat::Alac => 8,
            // MP3 knows only mono and stereo; Opus can do more with
            // multistream mapping family 1, which we do not implement.
            ExportFormat::Mp3 | ExportFormat::Opus => 2,
        }
    }
}

/// The wildcard of a save dialog that offers every export format, in the
/// order of [`ExportFormat::ALL`] (so a filter index maps to that list).
/// Each extension is listed in lower and upper case for GTK.
pub fn save_dialog_wildcard() -> String {
    ExportFormat::ALL
        .iter()
        .map(|f| {
            let e = f.extension();
            format!("{} (*.{e})|*.{e};*.{}", f.label(), e.to_uppercase())
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// Decides the format and final file name for a path picked in a save
/// dialog whose filter `filter` was selected: a known extension in the
/// name wins; otherwise the filter's format is used and its extension
/// appended (GTK does not add it by itself).
pub fn resolve_export_path(path: &Path, filter: Option<ExportFormat>) -> (PathBuf, ExportFormat) {
    if let Some(f) = ExportFormat::from_path(path) {
        return (path.to_path_buf(), f);
    }
    let f = filter.unwrap_or(ExportFormat::Wav);
    let mut name = path.as_os_str().to_owned();
    name.push(".");
    name.push(f.extension());
    (PathBuf::from(name), f)
}

/// What to write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExportOptions {
    /// The file format.
    pub format: ExportFormat,
    /// Target bit rate in kbit/s for the lossy formats (all channels
    /// together); `None` picks a transparent default. Ignored by lossless
    /// formats.
    pub bitrate_kbps: Option<u32>,
}

impl ExportOptions {
    /// Default quality for `format`.
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            bitrate_kbps: None,
        }
    }
}

/// A streaming encoder: feed it chunks, then finish it.
pub trait AudioWriter {
    /// Appends channel-major samples (all channels equally long, as many
    /// channels as the writer was created with).
    fn write(&mut self, channels: &[Vec<f32>]) -> Result<()>;
    /// Flushes the encoder and completes the file (headers, sizes, seek
    /// tables). Dropping a writer without finishing leaves a broken file.
    fn finish(self: Box<Self>) -> Result<()>;
}

impl AudioWriter for WavStreamWriter {
    fn write(&mut self, channels: &[Vec<f32>]) -> Result<()> {
        WavStreamWriter::write(self, channels)
    }

    fn finish(self: Box<Self>) -> Result<()> {
        WavStreamWriter::finish(*self)
    }
}

/// Creates `path` and returns a writer that encodes into it.
pub fn create_writer(
    path: &Path,
    options: ExportOptions,
    sample_rate: u32,
    channels: usize,
) -> Result<Box<dyn AudioWriter>> {
    let format = options.format;
    ensure!(sample_rate > 0, "sample rate is zero");
    ensure!(channels > 0, "recording has no channels");
    ensure!(
        channels <= format.max_channels(),
        "{} supports at most {} channels, the recording has {channels}",
        format.label(),
        format.max_channels()
    );
    if let Some(kbps) = options.bitrate_kbps {
        ensure!(
            (6..=512).contains(&kbps),
            "bit rate {kbps} kbit/s is out of range (6 to 512)"
        );
    }
    let open = || -> Result<BufWriter<File>> {
        Ok(BufWriter::new(
            File::create(path).with_context(|| format!("cannot write {}", path.display()))?,
        ))
    };
    Ok(match format {
        ExportFormat::Wav => Box::new(WavStreamWriter::create(path, sample_rate, channels as u16)?),
        ExportFormat::Flac => Box::new(FlacWriter::new(open()?, sample_rate, channels)?),
        ExportFormat::Vorbis => Box::new(VorbisWriter::new(open()?, sample_rate, channels, options.bitrate_kbps)?),
        ExportFormat::Mp3 => Box::new(Mp3Writer::new(open()?, sample_rate, channels, options.bitrate_kbps)?),
        ExportFormat::Opus => Box::new(OpusWriter::new(open()?, sample_rate, channels, options.bitrate_kbps)?),
        ExportFormat::Alac => Box::new(AlacWriter::new(open()?, sample_rate, channels)?),
    })
}

/// Checks that every channel of a chunk has the same length and that the
/// channel count matches; returns the length.
fn chunk_len(channels: &[Vec<f32>], expected: usize) -> Result<usize> {
    ensure!(
        channels.len() == expected,
        "chunk has {} channels, writer expects {expected}",
        channels.len()
    );
    let len = channels.first().map_or(0, Vec::len);
    ensure!(
        channels.iter().all(|c| c.len() == len),
        "channels of a chunk differ in length"
    );
    Ok(len)
}

/// A seed for Ogg stream serial numbers: they only have to differ between
/// streams that get chained into one file, so the clock is good enough.
fn ogg_serial() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0x6e6f_6973, |d| d.subsec_nanos() ^ d.as_secs() as u32)
}

// ---------------------------------------------------------------------------
// FLAC

/// Samples per FLAC frame; 4096 is the reference encoder's default for
/// 44.1/48 kHz and what `flacenc` is tuned for.
const FLAC_BLOCK: usize = 4096;
/// Bits per sample of exported FLAC files. 24 bits put the quantisation
/// noise ~146 dB below full scale, far under any microphone's noise, so no
/// dither is needed.
const FLAC_BITS: usize = 24;

/// Streaming FLAC encoder. `flacenc` encodes whole streams in memory, so
/// we drive its frame encoder ourselves: write a placeholder STREAMINFO,
/// append frames as blocks fill up, and rewrite STREAMINFO (total length,
/// MD5, frame sizes) in [`finish`](AudioWriter::finish).
struct FlacWriter {
    /// Output file.
    out: BufWriter<File>,
    /// Verified encoder settings.
    config: flacenc::error::Verified<flacenc::config::Encoder>,
    /// Header being filled in while frames are written.
    stream: flacenc::component::Stream,
    /// Interleaved integer samples waiting for a full block.
    pending: Vec<i32>,
    /// Frame buffer plus MD5/sample counting context, as `flacenc` wants it.
    buffers: (flacenc::source::FrameBuf, flacenc::source::Context),
    /// Number of channels.
    channels: usize,
}

impl FlacWriter {
    /// Writes the placeholder header.
    fn new(mut out: BufWriter<File>, sample_rate: u32, channels: usize) -> Result<Self> {
        use flacenc::error::Verify;
        let mut config = flacenc::config::Encoder::default();
        config.block_size = FLAC_BLOCK;
        // Frames are encoded one at a time; flacenc's thread pool is only
        // used by its whole-stream API.
        config.multithread = false;
        let config = config
            .into_verified()
            .map_err(|(_, e)| anyhow::anyhow!("FLAC encoder settings: {e:?}"))?;
        let mut stream = flacenc::component::Stream::new(sample_rate as usize, channels, FLAC_BITS)
            .map_err(|e| anyhow::anyhow!("FLAC cannot store this recording: {e:?}"))?;
        stream
            .stream_info_mut()
            .set_block_sizes(FLAC_BLOCK, FLAC_BLOCK)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        out.write_all(&flac_header(&stream)?)?;
        Ok(Self {
            out,
            config,
            stream,
            pending: Vec::with_capacity(FLAC_BLOCK * channels),
            buffers: (
                flacenc::source::FrameBuf::with_size(channels, FLAC_BLOCK).map_err(|e| anyhow::anyhow!("{e:?}"))?,
                flacenc::source::Context::new(FLAC_BITS, channels),
            ),
            channels,
        })
    }

    /// Encodes the first `frames` interleaved samples of `pending`.
    fn encode_block(&mut self, frames: usize) -> Result<()> {
        use flacenc::component::BitRepr;
        use flacenc::source::Fill;
        let n = frames * self.channels;
        // A short last block needs a frame buffer of its own size, or the
        // frame would be padded with stale samples.
        if frames != self.buffers.0.size() {
            self.buffers.0.resize(frames);
        }
        self.buffers
            .fill_interleaved(&self.pending[..n])
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let number = self.buffers.1.current_frame_number().unwrap_or(0);
        let frame = flacenc::encode_fixed_size_frame(&self.config, &self.buffers.0, number, self.stream.stream_info())
            .map_err(|e| anyhow::anyhow!("FLAC encoder: {e:?}"))?;
        self.stream.stream_info_mut().update_frame_info(&frame);
        let mut sink = flacenc::bitsink::ByteSink::new();
        frame.write(&mut sink).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        self.out.write_all(sink.as_slice())?;
        self.pending.drain(..n);
        Ok(())
    }
}

/// Serialises the `fLaC` marker and the STREAMINFO block (always 42 bytes).
fn flac_header(stream: &flacenc::component::Stream) -> Result<Vec<u8>> {
    use flacenc::component::BitRepr;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream.write(&mut sink).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(sink.as_slice().to_vec())
}

/// Converts a float sample to a signed integer of `bits` bits, clipping.
fn quantize(x: f32, bits: usize) -> i32 {
    let scale = (1i64 << (bits - 1)) as f64;
    let v = (f64::from(x) * scale).round();
    v.clamp(-scale, scale - 1.0) as i32
}

impl AudioWriter for FlacWriter {
    fn write(&mut self, channels: &[Vec<f32>]) -> Result<()> {
        let len = chunk_len(channels, self.channels)?;
        for i in 0..len {
            for c in channels {
                self.pending.push(quantize(c[i], FLAC_BITS));
            }
        }
        while self.pending.len() >= FLAC_BLOCK * self.channels {
            self.encode_block(FLAC_BLOCK)?;
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        let rest = self.pending.len() / self.channels;
        if rest > 0 {
            self.encode_block(rest)?;
        }
        let md5 = self.buffers.1.md5_digest();
        let total = self.buffers.1.total_samples();
        let info = self.stream.stream_info_mut();
        info.set_md5_digest(&md5);
        info.set_total_samples(total);
        // RFC 9639 8.2: the minimum block size excludes the last block,
        // which `update_frame_info` counted; a stream that is a single short
        // block states its real size in both fields.
        let block = if total < FLAC_BLOCK { total.max(16) } else { FLAC_BLOCK };
        info.set_block_sizes(block, block)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let header = flac_header(&self.stream)?;
        self.out.seek(SeekFrom::Start(0))?;
        self.out.write_all(&header)?;
        self.out.flush()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Ogg Vorbis

/// Default Vorbis quality (libvorbis scale -0.1 .. 1.0); 0.6 is `oggenc -q6`,
/// about 190 kbit/s for stereo and transparent for nearly all material.
const VORBIS_QUALITY: f32 = 0.6;

/// Streaming Ogg Vorbis encoder.
struct VorbisWriter {
    /// libvorbis wrapped by `vorbis_rs`; owns the output file.
    encoder: vorbis_rs::VorbisEncoder<BufWriter<File>>,
    /// Number of channels.
    channels: usize,
}

impl VorbisWriter {
    /// Sets up the encoder and writes the three header packets.
    fn new(out: BufWriter<File>, sample_rate: u32, channels: usize, kbps: Option<u32>) -> Result<Self> {
        use std::num::{NonZeroU32, NonZeroU8};
        let rate = NonZeroU32::new(sample_rate).context("sample rate is zero")?;
        let ch = u8::try_from(channels)
            .ok()
            .and_then(NonZeroU8::new)
            .context("unsupported channel count")?;
        let mut builder = vorbis_rs::VorbisEncoderBuilder::new_with_serial(rate, ch, out, ogg_serial() as i32);
        builder.bitrate_management_strategy(match kbps.and_then(|k| NonZeroU32::new(k * 1000)) {
            Some(target_bitrate) => vorbis_rs::VorbisBitrateManagementStrategy::Vbr { target_bitrate },
            None => vorbis_rs::VorbisBitrateManagementStrategy::QualityVbr {
                target_quality: VORBIS_QUALITY,
            },
        });
        builder.comment_tags([("ENCODER", "noiseextractor")])?;
        let encoder = builder.build().context("Vorbis encoder rejected the settings")?;
        Ok(Self { encoder, channels })
    }
}

impl AudioWriter for VorbisWriter {
    fn write(&mut self, channels: &[Vec<f32>]) -> Result<()> {
        let len = chunk_len(channels, self.channels)?;
        // libvorbis documents 1024 frames as a reasonable block size; huge
        // blocks only cost memory inside the library.
        for start in (0..len).step_by(1024) {
            let end = (start + 1024).min(len);
            let block: Vec<&[f32]> = channels.iter().map(|c| &c[start..end]).collect();
            self.encoder.encode_audio_block(&block)?;
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        let mut out = self.encoder.finish()?;
        out.flush()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MP3

/// Streaming MP3 encoder (LAME).
struct Mp3Writer {
    /// Output file.
    out: BufWriter<File>,
    /// LAME.
    encoder: mp3lame_encoder::Encoder,
    /// Number of channels (1 or 2).
    channels: usize,
    /// Reused output buffer.
    buf: Vec<u8>,
}

/// The LAME constant bit rate closest to `kbps` (MPEG-1 Layer III rates).
fn lame_bitrate(kbps: u32) -> mp3lame_encoder::Bitrate {
    use mp3lame_encoder::Bitrate::*;
    const RATES: [(u32, mp3lame_encoder::Bitrate); 16] = [
        (8, Kbps8),
        (16, Kbps16),
        (24, Kbps24),
        (32, Kbps32),
        (40, Kbps40),
        (48, Kbps48),
        (64, Kbps64),
        (80, Kbps80),
        (96, Kbps96),
        (112, Kbps112),
        (128, Kbps128),
        (160, Kbps160),
        (192, Kbps192),
        (224, Kbps224),
        (256, Kbps256),
        (320, Kbps320),
    ];
    RATES
        .iter()
        .min_by_key(|(r, _)| r.abs_diff(kbps))
        .map_or(Kbps192, |&(_, b)| b)
}

impl Mp3Writer {
    /// Sets up LAME: VBR quality 2 (`lame -V2`, ≈190 kbit/s stereo) unless
    /// a bit rate is given, which selects constant bit rate.
    fn new(out: BufWriter<File>, sample_rate: u32, channels: usize, kbps: Option<u32>) -> Result<Self> {
        use mp3lame_encoder::{Builder, Mode, Quality, VbrMode};
        let lame = |e: mp3lame_encoder::BuildError| anyhow::anyhow!("MP3 encoder: {e:?}");
        let mut b = Builder::new().context("cannot start the MP3 encoder")?;
        b.set_num_channels(channels as u8).map_err(lame)?;
        // LAME resamples by itself when the input rate is not an MPEG rate
        // (e.g. 96 kHz recordings become 48 kHz files).
        b.set_sample_rate(sample_rate).map_err(lame)?;
        b.set_mode(if channels == 1 { Mode::Mono } else { Mode::JointStereo })
            .map_err(lame)?;
        b.set_quality(Quality::NearBest).map_err(lame)?;
        match kbps {
            Some(k) => {
                b.set_vbr_mode(VbrMode::Off).map_err(lame)?;
                b.set_brate(lame_bitrate(k)).map_err(lame)?;
            }
            None => {
                b.set_vbr_mode(VbrMode::Mtrh).map_err(lame)?;
                b.set_vbr_quality(Quality::NearBest).map_err(lame)?;
            }
        }
        // The Xing/LAME info frame carries the exact length (encoder delay
        // and padding), which players need for gapless, exact-length
        // playback. It is written as a placeholder now and filled in by
        // `finish`.
        b.set_to_write_vbr_tag(true).map_err(lame)?;
        let encoder = b.build().map_err(lame)?;
        Ok(Self {
            out,
            encoder,
            channels,
            buf: Vec::new(),
        })
    }
}

impl AudioWriter for Mp3Writer {
    fn write(&mut self, channels: &[Vec<f32>]) -> Result<()> {
        use mp3lame_encoder::{DualPcm, MonoPcm};
        let len = chunk_len(channels, self.channels)?;
        for start in (0..len).step_by(8192) {
            let end = (start + 8192).min(len);
            self.buf.clear();
            self.buf.reserve(mp3lame_encoder::max_required_buffer_size(end - start));
            let r = if self.channels == 1 {
                self.encoder
                    .encode_to_vec(MonoPcm(&channels[0][start..end]), &mut self.buf)
            } else {
                self.encoder.encode_to_vec(
                    DualPcm {
                        left: &channels[0][start..end],
                        right: &channels[1][start..end],
                    },
                    &mut self.buf,
                )
            };
            r.map_err(|e| anyhow::anyhow!("MP3 encoder: {e:?}"))?;
            self.out.write_all(&self.buf)?;
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        self.buf.clear();
        // LAME's documentation asks for at least 7200 bytes for the flush.
        self.buf.reserve(7200);
        self.encoder
            .flush_to_vec::<mp3lame_encoder::FlushGap>(&mut self.buf)
            .map_err(|e| anyhow::anyhow!("MP3 encoder: {e:?}"))?;
        self.out.write_all(&self.buf)?;
        // Replace the placeholder info frame at the start of the file
        // (there is no ID3v2 tag before it).
        let mut tag = Vec::with_capacity(self.encoder.lame_tag_size());
        if self.encoder.lame_tag_encode_to_vec(&mut tag).is_some() {
            self.out.seek(SeekFrom::Start(self.encoder.id3v2_tag_size() as u64))?;
            self.out.write_all(&tag)?;
        }
        self.out.flush()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Opus

/// Opus always runs at 48 kHz here (decoders output 48 kHz anyway).
const OPUS_RATE: u32 = 48_000;
/// 20 ms frames: the size libopus is tuned for in music mode.
const OPUS_FRAME: usize = 960;

/// Fixed-ratio resampler for streaming input, built on `rubato`'s FFT
/// resampler, that removes the resampler's delay so output sample `k`
/// corresponds to input time `k / out_rate`.
struct Resampler {
    /// The FFT resampler.
    inner: rubato::FftFixedIn<f32>,
    /// Input waiting for a full chunk, per channel.
    pending: Vec<Vec<f32>>,
    /// Output frames still to drop because they are the filter's delay.
    skip: usize,
    /// Input frames seen.
    input_frames: u64,
    /// Output frames handed out.
    output_frames: u64,
    /// Input and output rate.
    rates: (u32, u32),
}

impl Resampler {
    /// Creates a resampler from `from` Hz to `to` Hz.
    fn new(from: u32, to: u32, channels: usize) -> Result<Self> {
        let inner = rubato::FftFixedIn::<f32>::new(from as usize, to as usize, 1024, 2, channels)
            .map_err(|e| anyhow::anyhow!("resampler: {e}"))?;
        let skip = rubato::Resampler::output_delay(&inner);
        Ok(Self {
            inner,
            pending: vec![Vec::new(); channels],
            skip,
            input_frames: 0,
            output_frames: 0,
            rates: (from, to),
        })
    }

    /// Feeds input; returns whatever output is ready (may be empty).
    fn push(&mut self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        use rubato::Resampler as _;
        for (p, c) in self.pending.iter_mut().zip(input) {
            p.extend_from_slice(c);
        }
        self.input_frames += input.first().map_or(0, Vec::len) as u64;
        let mut out = vec![Vec::new(); self.pending.len()];
        loop {
            let need = self.inner.input_frames_next();
            if self.pending[0].len() < need {
                break;
            }
            let res = self
                .inner
                .process(&self.pending, None)
                .map_err(|e| anyhow::anyhow!("resampler: {e}"))?;
            for p in &mut self.pending {
                p.drain(..need);
            }
            self.collect(res, &mut out, u64::MAX);
        }
        Ok(out)
    }

    /// Flushes the filter with silence and returns the remaining output,
    /// so that in total exactly `ceil(input · to / from)` frames came out.
    fn finish(&mut self) -> Result<Vec<Vec<f32>>> {
        use rubato::Resampler as _;
        let total = (self.input_frames * u64::from(self.rates.1)).div_ceil(u64::from(self.rates.0));
        let mut out = vec![Vec::new(); self.pending.len()];
        while self.output_frames < total {
            let need = self.inner.input_frames_next();
            for p in &mut self.pending {
                p.resize(need.max(p.len()), 0.0);
            }
            let res = self
                .inner
                .process(&self.pending, None)
                .map_err(|e| anyhow::anyhow!("resampler: {e}"))?;
            for p in &mut self.pending {
                p.drain(..need);
            }
            self.collect(res, &mut out, total);
        }
        Ok(out)
    }

    /// Appends resampler output to `out`, dropping the delay at the start
    /// and anything beyond `limit` frames in total.
    fn collect(&mut self, res: Vec<Vec<f32>>, out: &mut [Vec<f32>], limit: u64) {
        let n = res.first().map_or(0, Vec::len);
        let skip = self.skip.min(n);
        self.skip -= skip;
        let keep = ((n - skip) as u64).min(limit.saturating_sub(self.output_frames)) as usize;
        for (o, r) in out.iter_mut().zip(&res) {
            o.extend_from_slice(&r[skip..skip + keep]);
        }
        self.output_frames += keep as u64;
    }
}

/// Streaming Ogg Opus encoder following RFC 7845 (Ogg encapsulation).
struct OpusWriter {
    /// libopus.
    encoder: opus::Encoder,
    /// Ogg page writer around the output file.
    ogg: ogg::PacketWriter<'static, BufWriter<File>>,
    /// Ogg stream serial number.
    serial: u32,
    /// Resampler when the recording is not at 48 kHz.
    resampler: Option<Resampler>,
    /// Interleaved 48 kHz samples waiting for a full frame.
    pending: Vec<f32>,
    /// Number of channels (1 or 2).
    channels: usize,
    /// Encoder delay in 48 kHz samples; decoders drop that many samples
    /// from the start (`pre_skip` in the OpusHead packet).
    pre_skip: u64,
    /// 48 kHz samples of real audio received so far.
    samples: u64,
    /// 48 kHz samples encoded so far (including padding).
    encoded: u64,
    /// Encoded packet that has not been written yet: the last packet must
    /// be flagged as end of stream, which is only known in `finish`.
    held: Option<Vec<u8>>,
}

impl OpusWriter {
    /// Sets up libopus and writes the identification and comment headers.
    fn new(out: BufWriter<File>, sample_rate: u32, channels: usize, kbps: Option<u32>) -> Result<Self> {
        let op = |e: opus::Error| anyhow::anyhow!("Opus encoder: {e}");
        let ch = if channels == 1 {
            opus::Channels::Mono
        } else {
            opus::Channels::Stereo
        };
        let mut encoder = opus::Encoder::new(OPUS_RATE, ch, opus::Application::Audio).map_err(op)?;
        // 64 kbit/s per channel is transparent for most material with
        // Opus 1.3+; field recordings with lots of hiss benefit from the
        // headroom over the usual 96 kbit/s stereo.
        let bits = kbps.unwrap_or(64 * channels as u32) * 1000;
        encoder.set_bitrate(opus::Bitrate::Bits(bits as i32)).map_err(op)?;
        let pre_skip = u64::try_from(encoder.get_lookahead().map_err(op)?).unwrap_or(0);
        let serial = ogg_serial();
        let mut ogg = ogg::PacketWriter::new(out);
        // RFC 7845 5.1: identification header, alone on the first page.
        let mut head = b"OpusHead".to_vec();
        head.push(1); // version
        head.push(channels as u8);
        head.extend_from_slice(&(pre_skip as u16).to_le_bytes());
        head.extend_from_slice(&sample_rate.to_le_bytes()); // original rate, informational
        head.extend_from_slice(&0i16.to_le_bytes()); // output gain
        head.push(0); // channel mapping family 0: mono or stereo
        ogg.write_packet(head, serial, ogg::PacketWriteEndInfo::EndPage, 0)?;
        // RFC 7845 5.2: comment header, on its own page(s).
        let vendor = format!("noiseextractor (libopus {})", opus::version());
        let mut tags = b"OpusTags".to_vec();
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor.as_bytes());
        tags.extend_from_slice(&0u32.to_le_bytes()); // no user comments
        ogg.write_packet(tags, serial, ogg::PacketWriteEndInfo::EndPage, 0)?;
        Ok(Self {
            encoder,
            ogg,
            serial,
            resampler: if sample_rate == OPUS_RATE {
                None
            } else {
                Some(Resampler::new(sample_rate, OPUS_RATE, channels)?)
            },
            pending: Vec::new(),
            channels,
            pre_skip,
            samples: 0,
            encoded: 0,
            held: None,
        })
    }

    /// Appends 48 kHz audio and encodes all complete frames.
    fn push48(&mut self, channels: &[Vec<f32>]) -> Result<()> {
        let len = channels.first().map_or(0, Vec::len);
        for i in 0..len {
            for c in channels {
                self.pending.push(c[i]);
            }
        }
        self.samples += len as u64;
        while self.pending.len() >= OPUS_FRAME * self.channels {
            self.encode_frame()?;
        }
        Ok(())
    }

    /// Encodes the first frame of `pending` and writes the packet before it.
    fn encode_frame(&mut self) -> Result<()> {
        let n = OPUS_FRAME * self.channels;
        // 1275 bytes is the largest packet a single Opus frame can need
        // (RFC 6716 3.2.1); a little more does no harm.
        let packet = self
            .encoder
            .encode_vec_float(&self.pending[..n], 1500)
            .map_err(|e| anyhow::anyhow!("Opus encoder: {e}"))?;
        self.pending.drain(..n);
        if let Some(prev) = self.held.take() {
            // Granule position: 48 kHz samples decodable up to the end of
            // this packet, counting the pre-skip (RFC 7845 4).
            self.ogg
                .write_packet(prev, self.serial, ogg::PacketWriteEndInfo::NormalPacket, self.encoded)?;
        }
        self.encoded += OPUS_FRAME as u64;
        self.held = Some(packet);
        Ok(())
    }
}

impl AudioWriter for OpusWriter {
    fn write(&mut self, channels: &[Vec<f32>]) -> Result<()> {
        chunk_len(channels, self.channels)?;
        match self.resampler.as_mut() {
            Some(r) => {
                let out = r.push(channels)?;
                self.push48(&out)
            }
            None => self.push48(channels),
        }
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        if let Some(r) = self.resampler.as_mut() {
            let out = r.finish()?;
            self.push48(&out)?;
        }
        // Encode silence until the decoder, which runs `pre_skip` samples
        // behind, has produced every real sample.
        let wanted = self.pre_skip + self.samples;
        while self.encoded < wanted {
            self.pending.resize(OPUS_FRAME * self.channels, 0.0);
            self.encode_frame()?;
        }
        let end = self.pre_skip + self.samples;
        let last = match self.held.take() {
            Some(p) => p,
            // An empty recording still needs one packet to be a valid stream.
            None => {
                self.pending.resize(OPUS_FRAME * self.channels, 0.0);
                self.encode_frame()?;
                self.held.take().context("Opus encoder produced no packet")?
            }
        };
        // The end-of-stream granule position is smaller than the encoded
        // length; RFC 7845 4.4 tells decoders to trim the excess.
        self.ogg
            .write_packet(last, self.serial, ogg::PacketWriteEndInfo::EndStream, end)?;
        self.ogg.inner_mut().flush()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ALAC in MP4

/// Frames per ALAC packet (Apple's default).
const ALAC_FRAMES: usize = alac_encoder::DEFAULT_FRAMES_PER_PACKET as usize;

/// Small deterministic noise source for TPDF dither (xorshift32).
struct Dither(u32);

impl Dither {
    /// Uniform value in -0.5..0.5.
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) - 0.5
    }

    /// Converts to 16 bits with triangular dither of ±1 LSB, which turns
    /// the rounding error into constant, signal-independent noise instead
    /// of distortion that follows quiet sounds (decays, birds far away).
    fn quantize16(&mut self, x: f32) -> i16 {
        let d = self.uniform() + self.uniform();
        (x * 32768.0 + d).round().clamp(-32768.0, 32767.0) as i16
    }
}

/// Streaming ALAC encoder writing an MP4 (`.m4a`) file: `ftyp`, then the
/// packets in `mdat` as they are encoded, then `moov` with the sample
/// tables once the length is known.
struct AlacWriter {
    /// Output file.
    out: BufWriter<File>,
    /// Apple's encoder.
    encoder: alac_encoder::AlacEncoder,
    /// Describes the 16-bit interleaved input of `encoder`.
    input_format: alac_encoder::FormatDescription,
    /// Interleaved 16-bit little-endian PCM waiting for a full packet.
    pending: Vec<u8>,
    /// Output buffer for one packet.
    packet: Vec<u8>,
    /// Size of every packet written, for `stsz`.
    sizes: Vec<u32>,
    /// File offset of every packet, for `stco`/`co64`.
    offsets: Vec<u64>,
    /// File offset of the `mdat` box header.
    mdat_start: u64,
    /// Current write position.
    pos: u64,
    /// Number of channels.
    channels: usize,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Frames encoded.
    frames: u64,
    /// Dither for the 16-bit conversion.
    dither: Dither,
}

/// Appends an MP4 box (ISO/IEC 14496-12 4.2) with the given payload.
fn mp4_box(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(&((payload.len() + 8) as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
}

/// Appends a full box: version and 24-bit flags before the payload.
fn mp4_full_box(out: &mut Vec<u8>, kind: &[u8; 4], version: u8, flags: u32, payload: &[u8]) {
    let mut p = Vec::with_capacity(payload.len() + 4);
    p.push(version);
    p.extend_from_slice(&flags.to_be_bytes()[1..]);
    p.extend_from_slice(payload);
    mp4_box(out, kind, &p);
}

/// The identity transformation matrix of `mvhd` and `tkhd` (16.16 and 2.30
/// fixed point).
const MP4_MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

impl AlacWriter {
    /// Writes `ftyp` and opens `mdat`.
    fn new(mut out: BufWriter<File>, sample_rate: u32, channels: usize) -> Result<Self> {
        let mut head = Vec::new();
        let mut ftyp = Vec::new();
        ftyp.extend_from_slice(b"M4A ");
        ftyp.extend_from_slice(&0u32.to_be_bytes());
        for brand in [b"M4A ", b"mp42", b"isom"] {
            ftyp.extend_from_slice(brand);
        }
        mp4_box(&mut head, b"ftyp", &ftyp);
        // An 8-byte `free` box followed by a 32-bit `mdat` header: if the
        // file grows past 4 GiB, `finish` turns the pair into one 64-bit
        // `mdat` header in place (the trick ffmpeg uses).
        mp4_box(&mut head, b"free", &[]);
        let mdat_start = head.len() as u64;
        mp4_box(&mut head, b"mdat", &[]);
        out.write_all(&head)?;
        let output_format =
            alac_encoder::FormatDescription::alac(f64::from(sample_rate), ALAC_FRAMES as u32, channels as u32);
        let packet = vec![0; output_format.max_packet_size()];
        Ok(Self {
            out,
            encoder: alac_encoder::AlacEncoder::new(&output_format),
            input_format: alac_encoder::FormatDescription::pcm::<i16>(f64::from(sample_rate), channels as u32),
            pending: Vec::new(),
            packet,
            sizes: Vec::new(),
            offsets: Vec::new(),
            mdat_start,
            pos: head.len() as u64,
            channels,
            sample_rate,
            frames: 0,
            dither: Dither(0x2545_f491),
        })
    }

    /// Encodes the first `frames` frames of `pending` into one packet.
    fn encode_packet(&mut self, frames: usize) -> Result<()> {
        let n = frames * self.channels * 2;
        let size = self
            .encoder
            .encode(&self.input_format, &self.pending[..n], &mut self.packet);
        self.out.write_all(&self.packet[..size])?;
        self.offsets.push(self.pos);
        self.sizes.push(size as u32);
        self.pos += size as u64;
        self.frames += frames as u64;
        self.pending.drain(..n);
        Ok(())
    }

    /// Builds the `moov` box describing the packets written to `mdat`.
    fn moov(&self) -> Vec<u8> {
        let rate = self.sample_rate;
        let frames = self.frames;
        // Movie time scale in ms keeps mvhd/tkhd durations in 32 bits.
        let movie_duration = (frames * 1000 / u64::from(rate)).min(u64::from(u32::MAX)) as u32;
        let wide = frames > u64::from(u32::MAX);

        let mut mvhd = Vec::new();
        mvhd.extend_from_slice(&[0; 8]); // creation, modification time
        mvhd.extend_from_slice(&1000u32.to_be_bytes());
        mvhd.extend_from_slice(&movie_duration.to_be_bytes());
        mvhd.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
        mvhd.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
        mvhd.extend_from_slice(&[0; 10]);
        MP4_MATRIX.iter().for_each(|m| mvhd.extend_from_slice(&m.to_be_bytes()));
        mvhd.extend_from_slice(&[0; 24]); // pre_defined
        mvhd.extend_from_slice(&2u32.to_be_bytes()); // next track id

        let mut tkhd = Vec::new();
        tkhd.extend_from_slice(&[0; 8]);
        tkhd.extend_from_slice(&1u32.to_be_bytes()); // track id
        tkhd.extend_from_slice(&[0; 4]);
        tkhd.extend_from_slice(&movie_duration.to_be_bytes());
        tkhd.extend_from_slice(&[0; 8]);
        tkhd.extend_from_slice(&[0; 4]); // layer, alternate group
        tkhd.extend_from_slice(&0x0100u16.to_be_bytes()); // volume
        tkhd.extend_from_slice(&[0; 2]);
        MP4_MATRIX.iter().for_each(|m| tkhd.extend_from_slice(&m.to_be_bytes()));
        tkhd.extend_from_slice(&[0; 8]); // width, height

        // mdhd counts in samples; version 1 has 64-bit times for files
        // longer than 2^32 samples (a day at 48 kHz).
        let mut mdhd = Vec::new();
        if wide {
            mdhd.extend_from_slice(&[0; 16]);
            mdhd.extend_from_slice(&rate.to_be_bytes());
            mdhd.extend_from_slice(&frames.to_be_bytes());
        } else {
            mdhd.extend_from_slice(&[0; 8]);
            mdhd.extend_from_slice(&rate.to_be_bytes());
            mdhd.extend_from_slice(&(frames as u32).to_be_bytes());
        }
        mdhd.extend_from_slice(&0x55c4u16.to_be_bytes()); // language "und"
        mdhd.extend_from_slice(&[0; 2]);

        let mut hdlr = Vec::new();
        hdlr.extend_from_slice(&[0; 4]);
        hdlr.extend_from_slice(b"soun");
        hdlr.extend_from_slice(&[0; 12]);
        hdlr.extend_from_slice(b"SoundHandler\0");

        // Sample entry for ALAC (Apple's "ALAC in ISO base media" layout):
        // an AudioSampleEntry whose child `alac` box holds the encoder's
        // ALACSpecificConfig ("magic cookie").
        let mut entry = Vec::new();
        entry.extend_from_slice(&[0; 6]);
        entry.extend_from_slice(&1u16.to_be_bytes()); // data reference index
        entry.extend_from_slice(&[0; 8]); // version, revision, vendor
        entry.extend_from_slice(&(self.channels as u16).to_be_bytes());
        entry.extend_from_slice(&16u16.to_be_bytes()); // sample size
        entry.extend_from_slice(&[0; 4]); // compression id, packet size
                                          // 16.16 fixed point; rates above 65535 Hz do not fit, decoders
                                          // take the rate from the cookie then.
        let fixed = if rate <= 0xffff { rate << 16 } else { 0 };
        entry.extend_from_slice(&fixed.to_be_bytes());
        mp4_full_box(&mut entry, b"alac", 0, 0, &self.encoder.magic_cookie());
        let mut stsd = Vec::new();
        stsd.extend_from_slice(&1u32.to_be_bytes());
        mp4_box(&mut stsd, b"alac", &entry);

        // Every packet holds ALAC_FRAMES frames except possibly the last.
        let mut stts = Vec::new();
        let full = self.sizes.len() as u64 - u64::from(!frames.is_multiple_of(ALAC_FRAMES as u64));
        let mut runs: Vec<(u32, u32)> = Vec::new();
        if full > 0 {
            runs.push((full as u32, ALAC_FRAMES as u32));
        }
        if !frames.is_multiple_of(ALAC_FRAMES as u64) {
            runs.push((1, (frames % ALAC_FRAMES as u64) as u32));
        }
        stts.extend_from_slice(&(runs.len() as u32).to_be_bytes());
        for (count, delta) in runs {
            stts.extend_from_slice(&count.to_be_bytes());
            stts.extend_from_slice(&delta.to_be_bytes());
        }

        // One packet per chunk keeps stsc trivial; the offset table is
        // what grows instead.
        let mut stsc = Vec::new();
        stsc.extend_from_slice(&1u32.to_be_bytes());
        stsc.extend_from_slice(&1u32.to_be_bytes()); // first chunk
        stsc.extend_from_slice(&1u32.to_be_bytes()); // samples per chunk
        stsc.extend_from_slice(&1u32.to_be_bytes()); // sample description

        let mut stsz = Vec::new();
        stsz.extend_from_slice(&0u32.to_be_bytes()); // sizes differ
        stsz.extend_from_slice(&(self.sizes.len() as u32).to_be_bytes());
        self.sizes.iter().for_each(|s| stsz.extend_from_slice(&s.to_be_bytes()));

        let large = self.offsets.last().is_some_and(|&o| o > u64::from(u32::MAX));
        let mut stco = Vec::new();
        stco.extend_from_slice(&(self.offsets.len() as u32).to_be_bytes());
        for &o in &self.offsets {
            if large {
                stco.extend_from_slice(&o.to_be_bytes());
            } else {
                stco.extend_from_slice(&(o as u32).to_be_bytes());
            }
        }

        let mut stbl = Vec::new();
        mp4_full_box(&mut stbl, b"stsd", 0, 0, &stsd);
        mp4_full_box(&mut stbl, b"stts", 0, 0, &stts);
        mp4_full_box(&mut stbl, b"stsc", 0, 0, &stsc);
        mp4_full_box(&mut stbl, b"stsz", 0, 0, &stsz);
        mp4_full_box(&mut stbl, if large { b"co64" } else { b"stco" }, 0, 0, &stco);

        let mut dref = Vec::new();
        dref.extend_from_slice(&1u32.to_be_bytes());
        mp4_full_box(&mut dref, b"url ", 0, 1, &[]); // flag 1: data in this file
        let mut dinf = Vec::new();
        mp4_full_box(&mut dinf, b"dref", 0, 0, &dref);

        let mut minf = Vec::new();
        mp4_full_box(&mut minf, b"smhd", 0, 0, &[0; 4]);
        mp4_box(&mut minf, b"dinf", &dinf);
        mp4_box(&mut minf, b"stbl", &stbl);

        let mut mdia = Vec::new();
        mp4_full_box(&mut mdia, b"mdhd", u8::from(wide), 0, &mdhd);
        mp4_full_box(&mut mdia, b"hdlr", 0, 0, &hdlr);
        mp4_box(&mut mdia, b"minf", &minf);

        let mut trak = Vec::new();
        mp4_full_box(&mut trak, b"tkhd", 0, 7, &tkhd); // enabled, in movie, in preview
        mp4_box(&mut trak, b"mdia", &mdia);

        let mut moov_payload = Vec::new();
        mp4_full_box(&mut moov_payload, b"mvhd", 0, 0, &mvhd);
        mp4_box(&mut moov_payload, b"trak", &trak);
        let mut moov = Vec::new();
        mp4_box(&mut moov, b"moov", &moov_payload);
        moov
    }
}

impl AudioWriter for AlacWriter {
    fn write(&mut self, channels: &[Vec<f32>]) -> Result<()> {
        let len = chunk_len(channels, self.channels)?;
        for i in 0..len {
            for c in channels {
                let s = self.dither.quantize16(c[i]);
                self.pending.extend_from_slice(&s.to_le_bytes());
            }
        }
        let packet_bytes = ALAC_FRAMES * self.channels * 2;
        while self.pending.len() >= packet_bytes {
            self.encode_packet(ALAC_FRAMES)?;
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        let rest = self.pending.len() / (self.channels * 2);
        if rest > 0 {
            self.encode_packet(rest)?;
        }
        let mdat_size = self.pos - self.mdat_start;
        if let Ok(size) = u32::try_from(mdat_size) {
            self.out.seek(SeekFrom::Start(self.mdat_start))?;
            self.out.write_all(&size.to_be_bytes())?;
        } else {
            // Grow the header into the preceding `free` box: size 1 means
            // "64-bit size follows the type".
            let start = self.mdat_start - 8;
            self.out.seek(SeekFrom::Start(start))?;
            self.out.write_all(&1u32.to_be_bytes())?;
            self.out.write_all(b"mdat")?;
            self.out.write_all(&(self.pos - start).to_be_bytes())?;
        }
        self.out.seek(SeekFrom::Start(self.pos))?;
        let moov = self.moov();
        self.out.write_all(&moov)?;
        self.out.flush()?;
        Ok(())
    }
}

/// Fails with a helpful message when a file name has an extension we cannot
/// write; used by the command line.
pub fn format_for_output(path: &Path, explicit: Option<&str>) -> Result<ExportFormat> {
    if let Some(name) = explicit {
        return ExportFormat::from_name(name).with_context(|| {
            format!(
                "unknown format {name:?}; known: {}",
                ExportFormat::ALL.map(ExportFormat::name).join(", ")
            )
        });
    }
    match ExportFormat::from_path(path) {
        Some(f) => Ok(f),
        None => bail!(
            "cannot tell the format from {}; use one of the extensions {} or --format",
            path.display(),
            ExportFormat::ALL.map(|f| format!(".{}", f.extension())).join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioData;

    /// Two seconds of a stereo test signal: 1 kHz left, 1.5 kHz right, so
    /// swapped or mixed channels are noticed.
    fn tones(rate: u32, seconds: f32) -> Vec<Vec<f32>> {
        let n = (rate as f32 * seconds) as usize;
        let tone = |f: f32, a: f32| {
            (0..n)
                .map(|i| a * (2.0 * std::f32::consts::PI * f * i as f32 / rate as f32).sin())
                .collect::<Vec<_>>()
        };
        vec![tone(1000.0, 0.5), tone(1500.0, 0.3)]
    }

    /// Amplitude of the `f` Hz component of `x` (a single DFT bin).
    fn amplitude(x: &[f32], f: f32, rate: u32) -> f32 {
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for (i, &v) in x.iter().enumerate() {
            let w = 2.0 * std::f64::consts::PI * f64::from(f) * i as f64 / f64::from(rate);
            s += f64::from(v) * w.sin();
            c += f64::from(v) * w.cos();
        }
        (2.0 * (s * s + c * c).sqrt() / x.len() as f64) as f32
    }

    /// Writes `channels` in uneven chunks, as the processing engine does.
    fn export(path: &Path, options: ExportOptions, rate: u32, channels: &[Vec<f32>]) {
        let mut w = create_writer(path, options, rate, channels.len()).unwrap();
        let len = channels[0].len();
        let mut start = 0;
        for size in [1, 777, 4096, 5000, 10_000].into_iter().cycle() {
            if start >= len {
                break;
            }
            let end = (start + size).min(len);
            let chunk: Vec<Vec<f32>> = channels.iter().map(|c| c[start..end].to_vec()).collect();
            w.write(&chunk).unwrap();
            start = end;
        }
        w.finish().unwrap();
    }

    /// Checks that a decoded lossy file still has both tones on the right
    /// channels (away from the edges, where encoders fade in).
    fn assert_tones(decoded: &[impl AsRef<[f32]>], rate: u32) {
        let (left, right) = (decoded[0].as_ref(), decoded[1].as_ref());
        let mid = &left[rate as usize / 4..left.len() - rate as usize / 4];
        let right = &right[rate as usize / 4..right.len() - rate as usize / 4];
        let (l1, l2) = (amplitude(mid, 1000.0, rate), amplitude(mid, 1500.0, rate));
        let (r1, r2) = (amplitude(right, 1000.0, rate), amplitude(right, 1500.0, rate));
        assert!((l1 - 0.5).abs() < 0.05, "left 1 kHz amplitude {l1}");
        assert!((r2 - 0.3).abs() < 0.03, "right 1.5 kHz amplitude {r2}");
        assert!(l2 < 0.01 && r1 < 0.01, "channels leak: {l2} {r1}");
    }

    #[test]
    fn formats_round_trip_through_names_and_extensions() {
        for f in ExportFormat::ALL {
            assert_eq!(ExportFormat::from_name(f.name()), Some(f));
            assert_eq!(ExportFormat::from_name(&f.extension().to_uppercase()), Some(f));
            let p = PathBuf::from(format!("x.{}", f.extension()));
            assert_eq!(resolve_export_path(&p, None), (p.clone(), f));
        }
        assert_eq!(ExportFormat::from_name("oga"), Some(ExportFormat::Vorbis));
        assert_eq!(ExportFormat::from_name("aac"), None);
    }

    #[test]
    fn missing_extension_comes_from_the_dialog_filter() {
        let (p, f) = resolve_export_path(Path::new("/tmp/owl"), Some(ExportFormat::Opus));
        assert_eq!((p, f), (PathBuf::from("/tmp/owl.opus"), ExportFormat::Opus));
        // A typed extension wins over the filter.
        let (p, f) = resolve_export_path(Path::new("owl.flac"), Some(ExportFormat::Mp3));
        assert_eq!((p, f), (PathBuf::from("owl.flac"), ExportFormat::Flac));
        // Unknown extensions are kept and the filter's is appended.
        let (p, _) = resolve_export_path(Path::new("owl.v2"), Some(ExportFormat::Wav));
        assert_eq!(p, PathBuf::from("owl.v2.wav"));
    }

    #[test]
    fn wildcard_lists_formats_in_order() {
        let w = save_dialog_wildcard();
        let parts: Vec<&str> = w.split('|').collect();
        assert_eq!(parts.len(), 2 * ExportFormat::ALL.len());
        for (i, f) in ExportFormat::ALL.iter().enumerate() {
            assert!(parts[2 * i + 1].contains(&format!("*.{}", f.extension())));
            assert!(parts[2 * i + 1].contains(&format!("*.{}", f.extension().to_uppercase())));
        }
    }

    #[test]
    fn command_line_format_choice() {
        assert_eq!(format_for_output(Path::new("a.mp3"), None).unwrap(), ExportFormat::Mp3);
        assert_eq!(
            format_for_output(Path::new("a.bin"), Some("opus")).unwrap(),
            ExportFormat::Opus
        );
        assert!(format_for_output(Path::new("a.bin"), None).is_err());
        assert!(format_for_output(Path::new("a.wav"), Some("aiff")).is_err());
    }

    #[test]
    fn flac_is_lossless_at_24_bits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.flac");
        let mut ch = tones(44_100, 0.5);
        // Values beyond full scale must clip, not wrap around.
        ch[0][100] = 1.5;
        ch[1][100] = -1.5;
        export(&path, ExportOptions::new(ExportFormat::Flac), 44_100, &ch);
        let back = AudioData::load(&path).unwrap();
        assert_eq!(back.sample_rate, 44_100);
        assert_eq!(back.channels.len(), 2);
        assert_eq!(back.len(), ch[0].len());
        for (orig, dec) in ch.iter().zip(&back.channels) {
            for (&o, &d) in orig.iter().zip(dec) {
                assert_eq!(d, quantize(o, 24) as f32 / 8_388_608.0);
            }
        }
    }

    #[test]
    fn flac_shorter_than_one_block() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short.flac");
        let ch = vec![vec![0.25f32; 300]];
        export(&path, ExportOptions::new(ExportFormat::Flac), 8000, &ch);
        let back = AudioData::load(&path).unwrap();
        assert_eq!(back.len(), 300);
        assert!(back.channels[0].iter().all(|&s| (s - 0.25).abs() < 1e-6));
    }

    #[test]
    fn alac_round_trip_within_dither() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.m4a");
        let ch = tones(48_000, 0.5);
        export(&path, ExportOptions::new(ExportFormat::Alac), 48_000, &ch);
        let back = AudioData::load(&path).unwrap();
        assert_eq!(back.sample_rate, 48_000);
        assert_eq!(back.channels.len(), 2);
        assert_eq!(back.len(), ch[0].len());
        for (orig, dec) in ch.iter().zip(&back.channels) {
            for (&o, &d) in orig.iter().zip(dec) {
                // Rounding plus triangular dither stay within 1.5 LSB.
                assert!((o - d).abs() <= 1.5 / 32768.0, "{o} vs {d}");
            }
        }
    }

    #[test]
    fn vorbis_keeps_length_and_tones() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.ogg");
        let ch = tones(44_100, 1.0);
        export(&path, ExportOptions::new(ExportFormat::Vorbis), 44_100, &ch);
        // Decoded with libvorbis rather than Symphonia: Symphonia 0.5 does
        // not apply the end trimming of the last granule position, so it
        // returns up to one block too many.
        let mut dec = vorbis_rs::VorbisDecoder::new(File::open(&path).unwrap()).unwrap();
        assert_eq!(dec.sampling_frequency().get(), 44_100);
        let mut back = vec![Vec::new(); usize::from(dec.channels().get())];
        while let Some(block) = dec.decode_audio_block().unwrap() {
            for (b, s) in back.iter_mut().zip(block.samples()) {
                b.extend_from_slice(s);
            }
        }
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].len(), ch[0].len());
        assert_tones(&back, 44_100);
        // Symphonia, which the program opens files with, still decodes it.
        assert_tones(&AudioData::load(&path).unwrap().channels, 44_100);
    }

    #[test]
    fn vorbis_with_a_bit_rate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.ogg");
        let ch = tones(48_000, 1.0);
        let options = ExportOptions {
            format: ExportFormat::Vorbis,
            bitrate_kbps: Some(96),
        };
        export(&path, options, 48_000, &ch);
        let bytes = std::fs::metadata(&path).unwrap().len();
        // One second at 96 kbit/s is 12 kB; allow for headers and VBR.
        assert!(bytes < 24_000, "{bytes} bytes");
        assert_tones(&AudioData::load(&path).unwrap().channels, 48_000);
    }

    #[test]
    fn mp3_keeps_tones() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.mp3");
        let ch = tones(44_100, 1.0);
        export(&path, ExportOptions::new(ExportFormat::Mp3), 44_100, &ch);
        let back = AudioData::load(&path).unwrap();
        assert_eq!(back.sample_rate, 44_100);
        // Without gapless trimming the decoder adds the encoder delay and
        // padding, but never more than two frames' worth.
        assert!(back.len() >= ch[0].len() && back.len() < ch[0].len() + 2 * 1152 + 1105);
        assert_tones(&back.channels, 44_100);
    }

    #[test]
    fn mp3_constant_bit_rate_and_high_sample_rate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.mp3");
        let ch = tones(96_000, 1.0);
        let options = ExportOptions {
            format: ExportFormat::Mp3,
            bitrate_kbps: Some(128),
        };
        export(&path, options, 96_000, &ch);
        let back = AudioData::load(&path).unwrap();
        // MP3 has no 96 kHz; LAME resamples.
        assert_eq!(back.sample_rate, 48_000);
        assert_tones(&back.channels, 48_000);
        let bytes = std::fs::metadata(&path).unwrap().len();
        assert!((14_000..20_000).contains(&bytes), "{bytes} bytes for 1 s at 128 kbit/s");
    }

    #[test]
    fn mp3_and_opus_refuse_surround() {
        let dir = tempfile::tempdir().unwrap();
        for f in [ExportFormat::Mp3, ExportFormat::Opus] {
            let path = dir.path().join(format!("t.{}", f.extension()));
            let err = create_writer(&path, ExportOptions::new(f), 48_000, 3).err().unwrap();
            assert!(format!("{err}").contains("at most 2 channels"), "{err}");
        }
    }

    /// Decodes an Ogg Opus file with libopus, honouring pre-skip and the
    /// end trimming of the last granule position, as players do.
    fn decode_opus(path: &Path) -> (u64, Vec<Vec<f32>>) {
        let mut r = ogg::PacketReader::new(std::io::BufReader::new(File::open(path).unwrap()));
        let head = r.read_packet_expected().unwrap();
        assert_eq!(&head.data[..8], b"OpusHead");
        let channels = usize::from(head.data[9]);
        let pre_skip = u64::from(u16::from_le_bytes([head.data[10], head.data[11]]));
        let tags = r.read_packet_expected().unwrap();
        assert_eq!(&tags.data[..8], b"OpusTags");
        let ch = if channels == 1 {
            opus::Channels::Mono
        } else {
            opus::Channels::Stereo
        };
        let mut dec = opus::Decoder::new(48_000, ch).unwrap();
        let mut pcm = Vec::new();
        let mut end = 0;
        while let Some(p) = r.read_packet().unwrap() {
            let mut buf = vec![0f32; 5760 * channels];
            let n = dec.decode_float(&p.data, &mut buf, false).unwrap();
            pcm.extend_from_slice(&buf[..n * channels]);
            if p.last_in_stream() {
                end = p.absgp_page();
            }
        }
        let frames: Vec<Vec<f32>> = (0..channels)
            .map(|c| {
                pcm.chunks(channels)
                    .map(|f| f[c])
                    .skip(pre_skip as usize)
                    .take((end - pre_skip) as usize)
                    .collect()
            })
            .collect();
        (pre_skip, frames)
    }

    #[test]
    fn opus_at_48k_keeps_exact_length_and_tones() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.opus");
        let ch = tones(48_000, 1.0);
        export(&path, ExportOptions::new(ExportFormat::Opus), 48_000, &ch);
        let (pre_skip, back) = decode_opus(&path);
        assert!(pre_skip > 0);
        assert_eq!(back[0].len(), ch[0].len());
        assert_tones(&back, 48_000);
        // The encoder delay is compensated: the decoded tone is in phase
        // with the original.
        let err: f32 = back[0][4800..43_200]
            .iter()
            .zip(&ch[0][4800..43_200])
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 38_400.0;
        assert!(err.sqrt() < 0.05, "rms error {}", err.sqrt());
    }

    #[test]
    fn opus_resamples_other_rates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.opus");
        let ch = tones(44_100, 1.0);
        export(&path, ExportOptions::new(ExportFormat::Opus), 44_100, &ch);
        let (_, back) = decode_opus(&path);
        assert_eq!(back[0].len(), 48_000);
        assert_tones(&back, 48_000);
    }

    #[test]
    fn opus_mono_and_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.opus");
        export(&path, ExportOptions::new(ExportFormat::Opus), 48_000, &[vec![0.1; 100]]);
        let (_, back) = decode_opus(&path);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].len(), 100);
        let empty = dir.path().join("e.opus");
        create_writer(&empty, ExportOptions::new(ExportFormat::Opus), 48_000, 2)
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(decode_opus(&empty).1[0].len(), 0);
    }

    #[test]
    fn resampler_has_exact_length_and_no_delay() {
        let input = tones(44_100, 0.5);
        let mut r = Resampler::new(44_100, 48_000, 2).unwrap();
        let mut out = vec![Vec::new(); 2];
        for chunk in input[0].chunks(333).zip(input[1].chunks(333)) {
            let o = r.push(&[chunk.0.to_vec(), chunk.1.to_vec()]).unwrap();
            for (a, b) in out.iter_mut().zip(o) {
                a.extend(b);
            }
        }
        for (a, b) in out.iter_mut().zip(r.finish().unwrap()) {
            a.extend(b);
        }
        assert_eq!(out[0].len(), 24_000);
        let expected = tones(48_000, 0.5);
        for i in 1000..23_000 {
            assert!((out[0][i] - expected[0][i]).abs() < 1e-3, "sample {i}");
        }
    }
}
