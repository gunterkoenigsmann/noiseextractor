//! `noiseextractor-cli`: runs noiseextractor filter projects without a GUI.
//!
//! Useful for batch processing, for scripting, and for testing the same
//! processing code the GUI uses. Every subcommand reads an audio file; the
//! filters come from a project file saved by the GUI (or by `mark` /
//! `denoise` below).
//!
//! ```text
//! noiseextractor-cli denoise in.flac -o clean.wav --save-project p.json
//! noiseextractor-cli mark in.flac --time 12.3:13.1 --freq 2000:4500 \
//!     --name "blackbird" --extract --project p.json --save-project p.json
//! noiseextractor-cli process in.flac --project p.json -o out.wav
//! noiseextractor-cli spectrogram in.flac --project p.json -o view.png
//! ```

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use noiseextractor_core::analysis::Selection;
use noiseextractor_core::audio::WavStreamWriter;
use noiseextractor_core::engine::{process_all, ChunkLayout, ChunkResult};
use noiseextractor_core::filter::{Filter, FilterAction, FilterKind, Project};
use noiseextractor_core::noise::{mean_power, GainRule, NoiseEstimate};
use noiseextractor_core::render::{render_rgb, View};
use noiseextractor_core::stft::Stft;
use noiseextractor_core::AudioData;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Command-line arguments.
#[derive(Parser)]
#[command(name = "noiseextractor-cli", version, about = "Suppress or extract noises in audio files using noiseextractor projects")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Worker threads (default: number of CPU cores).
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// STFT frames per processing chunk.
    #[arg(long, global = true, default_value_t = 256)]
    chunk_frames: usize,
}

/// Subcommands.
#[derive(Subcommand)]
enum Command {
    /// Apply all enabled filters of a project and write a 32-bit float WAV.
    Process {
        /// Input audio file.
        input: PathBuf,
        /// Project file (default: background noise reduction only).
        #[arg(long)]
        project: Option<PathBuf>,
        /// Output WAV file.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Background noise reduction with explicit parameters.
    Denoise {
        /// Input audio file.
        input: PathBuf,
        /// Output WAV file.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Maximum attenuation in dB.
        #[arg(long, default_value_t = 18.0)]
        reduction_db: f32,
        /// Gain rule.
        #[arg(long, value_enum, default_value_t = Rule::Lsa)]
        rule: Rule,
        /// Learn the noise spectrum from a noise-only stretch, "START:END" in
        /// seconds, instead of estimating it adaptively.
        #[arg(long)]
        noise_from: Option<String>,
        /// Also save the resulting project.
        #[arg(long)]
        save_project: Option<PathBuf>,
    },
    /// Mark a region, report what belongs to it, and add it as a filter.
    Mark {
        /// Input audio file.
        input: PathBuf,
        /// Time range "START:END" in seconds.
        #[arg(long)]
        time: String,
        /// Frequency range "LOW:HIGH" in Hz.
        #[arg(long)]
        freq: String,
        /// Name of the new filter.
        #[arg(long, default_value = "Marked noise")]
        name: String,
        /// Extract instead of suppress.
        #[arg(long)]
        extract: bool,
        /// Project to add the filter to (default: a new project).
        #[arg(long)]
        project: Option<PathBuf>,
        /// Where to save the extended project.
        #[arg(long)]
        save_project: Option<PathBuf>,
    },
    /// Render the processed spectrogram (with the GUI's colours) as PNG.
    Spectrogram {
        /// Input audio file.
        input: PathBuf,
        /// Project file (default: no filters).
        #[arg(long)]
        project: Option<PathBuf>,
        /// Output PNG file.
        #[arg(short, long)]
        output: PathBuf,
        /// Image width in pixels.
        #[arg(long, default_value_t = 1200)]
        width: u32,
        /// Image height in pixels.
        #[arg(long, default_value_t = 400)]
        height: u32,
        /// Highest frequency shown (default: Nyquist).
        #[arg(long)]
        max_hz: Option<f32>,
        /// Logarithmic frequency axis.
        #[arg(long)]
        log: bool,
    },
    /// Write a project with the default filters, as a starting point.
    NewProject {
        /// Output project file.
        output: PathBuf,
    },
}

/// Gain rule choices on the command line.
#[derive(Clone, Copy, ValueEnum)]
enum Rule {
    /// Wiener gain.
    Wiener,
    /// MMSE log-spectral amplitude.
    Lsa,
}

/// Entry point: parse arguments and report errors in one line.
fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

/// Executes the chosen subcommand.
fn run(cli: Cli) -> Result<()> {
    let threads = cli.threads.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    anyhow::ensure!(cli.chunk_frames >= 1, "--chunk-frames must be at least 1");
    match cli.command {
        Command::Process { input, project, output } => {
            let audio = load(&input)?;
            let mut project = match project {
                Some(p) => Project::load(&p)?,
                None => Project::default(),
            };
            project.prepare(&audio);
            write_processed(&audio, &project, cli.chunk_frames, threads, &output)
        }
        Command::Denoise { input, output, reduction_db, rule, noise_from, save_project } => {
            let audio = load(&input)?;
            let mut project = Project::default();
            let FilterKind::NoiseReduction(nr) = &mut project.filters[0].kind else { unreachable!("default project starts with noise reduction") };
            nr.reduction_db = reduction_db;
            nr.rule = match rule {
                Rule::Wiener => GainRule::Wiener,
                Rule::Lsa => GainRule::MmseLsa,
            };
            if let Some(range) = noise_from {
                let (a, b) = parse_range(&range)?;
                nr.estimate = NoiseEstimate::Profile(learn_noise(&audio, &project.stft, a, b)?);
            }
            if let Some(p) = save_project {
                project.save(&p)?;
            }
            match output {
                Some(o) => write_processed(&audio, &project, cli.chunk_frames, threads, &o),
                None => Ok(()),
            }
        }
        Command::Mark { input, time, freq, name, extract, project, save_project } => {
            let audio = load(&input)?;
            let mut project = match project {
                Some(p) => Project::load(&p)?,
                None => Project::default(),
            };
            let (start_s, end_s) = parse_range(&time)?;
            let (low, high) = parse_range(&freq)?;
            let selection = Selection { start_s, end_s, low_hz: low as f32, high_hz: high as f32 };
            let action = if extract { FilterAction::Extract } else { FilterAction::Suppress };
            project.filters.push(Filter::event(name, selection, action));
            project.prepare(&audio);
            print_report(&project, audio.sample_rate);
            if let Some(p) = save_project {
                project.save(&p)?;
            }
            Ok(())
        }
        Command::Spectrogram { input, project, output, width, height, max_hz, log } => {
            let audio = load(&input)?;
            let mut project = match project {
                Some(p) => Project::load(&p)?,
                None => Project { filters: vec![], ..Project::default() },
            };
            project.prepare(&audio);
            write_spectrogram(&audio, &project, cli.chunk_frames, threads, &output, width, height, max_hz, log)
        }
        Command::NewProject { output } => Project::default().save(&output),
    }
}

/// Loads an audio file, refusing empty ones.
fn load(path: &Path) -> Result<AudioData> {
    let audio = AudioData::load(path)?;
    if audio.is_empty() {
        bail!("{} contains no samples", path.display());
    }
    eprintln!("{}: {} ch, {} Hz, {:.1} s", path.display(), audio.channels.len(), audio.sample_rate, audio.duration());
    Ok(audio)
}

/// Parses "A:B" into two numbers.
fn parse_range(s: &str) -> Result<(f64, f64)> {
    let (a, b) = s.split_once(':').with_context(|| format!("expected START:END, got {s:?}"))?;
    let a: f64 = a.trim().parse().with_context(|| format!("not a number: {a:?}"))?;
    let b: f64 = b.trim().parse().with_context(|| format!("not a number: {b:?}"))?;
    Ok((a.min(b), a.max(b)))
}

/// Mean noise power spectrum of the stretch `start_s .. end_s`.
fn learn_noise(audio: &AudioData, cfg: &noiseextractor_core::StftConfig, start_s: f64, end_s: f64) -> Result<Vec<f32>> {
    let stft = Stft::new(*cfg);
    let a = (start_s * audio.sample_rate as f64 / cfg.hop as f64) as i64;
    let b = (end_s * audio.sample_rate as f64 / cfg.hop as f64) as i64;
    anyhow::ensure!(b > a, "noise range is empty");
    let power = stft.analyze(&audio.mono(), a, (b - a) as usize).power();
    Ok(mean_power(&power, cfg.bins()))
}

/// Processes everything and streams the result into a WAV file.
fn write_processed(audio: &AudioData, project: &Project, chunk_frames: usize, threads: usize, output: &Path) -> Result<()> {
    let mut writer = WavStreamWriter::create(output, audio.sample_rate, audio.channels.len() as u16)?;
    let total = audio.len().max(1);
    let mut last_percent = usize::MAX;
    process_all(audio, project, chunk_frames, threads, |r| {
        let percent = (r.start_sample + r.audio[0].len()) * 100 / total;
        if percent / 10 != last_percent / 10 {
            eprintln!("{percent:3}%");
            last_percent = percent;
        }
        writer.write(&r.audio)
    })?;
    writer.finish()?;
    eprintln!("wrote {}", output.display());
    Ok(())
}

/// Prints what the analysis found for each event filter.
fn print_report(project: &Project, sample_rate: u32) {
    for f in &project.filters {
        let FilterKind::Event(e) = &f.kind else { continue };
        let Some(a) = &e.analysis else { continue };
        let track: Vec<f32> = a.pitch_track.iter().flatten().map(|&b| project.stft.bin_to_hz(b, sample_rate)).collect();
        println!("filter {:?} ({:?})", f.name, f.action);
        if let (Some(lo), Some(hi)) = (track.iter().cloned().reduce(f32::min), track.iter().cloned().reduce(f32::max)) {
            println!("  dominant frequency in selection: {lo:.0} .. {hi:.0} Hz over {} frames", track.len());
        } else {
            println!("  nothing above the noise floor in the selection");
        }
        for p in &a.partials {
            println!("  related partial at {:.3} x pitch (score {:.2})", p.ratio, p.score);
        }
        let comod = (0..a.bins).filter(|&k| a.comodulated.iter().any(|f| f[k] > 0.5)).count();
        println!("  co-modulated bins: {comod}");
    }
}

/// Processes the file and renders the spectrogram into a PNG.
#[allow(clippy::too_many_arguments)]
fn write_spectrogram(
    audio: &AudioData,
    project: &Project,
    chunk_frames: usize,
    threads: usize,
    output: &Path,
    width: u32,
    height: u32,
    max_hz: Option<f32>,
    log: bool,
) -> Result<()> {
    anyhow::ensure!(width > 0 && height > 0, "image must not be empty");
    // Keep only the display layers; the audio is not needed for a picture.
    let mut chunks: HashMap<usize, Arc<ChunkResult>> = HashMap::new();
    process_all(audio, project, chunk_frames, threads, |r| {
        let slim = ChunkResult { audio: Vec::new(), ..clone_layers(r) };
        chunks.insert(r.index, Arc::new(slim));
        Ok(())
    })?;
    let layout = ChunkLayout { hop: project.stft.hop, chunk_frames, total_samples: audio.len() };
    let view = View {
        start_sample: 0.0,
        samples_per_px: audio.len() as f64 / width as f64,
        low_hz: if log { 50.0 } else { 0.0 },
        high_hz: max_hz.unwrap_or(audio.sample_rate as f32 / 2.0),
        log_freq: log,
        db_min: -100.0,
        db_max: -10.0,
    };
    let rgb = render_rgb(&view, width, height, layout, audio.sample_rate, project.stft.fft_size, &|c| chunks.get(&c).cloned());
    let file = std::fs::File::create(output).with_context(|| format!("cannot write {}", output.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&rgb)?;
    eprintln!("wrote {}", output.display());
    Ok(())
}

/// Copies the display layers of a chunk result (they are not `Clone` on
/// purpose, to avoid accidental copies of the audio in the engine).
fn clone_layers(r: &ChunkResult) -> ChunkResult {
    ChunkResult {
        index: r.index,
        generation: r.generation,
        first_frame: r.first_frame,
        frames: r.frames,
        bins: r.bins,
        start_sample: r.start_sample,
        audio: Vec::new(),
        original_db: r.original_db.clone(),
        output_db: r.output_db.clone(),
        selected: r.selected.clone(),
        inferred: r.inferred.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_range;

    #[test]
    fn ranges_parse_in_either_order() {
        assert_eq!(parse_range("1.5:0.5").unwrap(), (0.5, 1.5));
        assert!(parse_range("1.5").is_err());
        assert!(parse_range("a:1").is_err());
    }
}
