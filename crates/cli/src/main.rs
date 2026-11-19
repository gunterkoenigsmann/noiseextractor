//! `noiseextractor-cli`: runs noiseextractor filter projects without a GUI.
//!
//! Useful for batch processing, for scripting, and for testing the same
//! processing code the GUI uses. Every subcommand reads an audio file; the
//! filters come from a project file saved by the GUI (or by `mark` /
//! `denoise` below).
//!
//! ```text
//! noiseextractor-cli denoise in.flac -o clean.wav --save-project birds.nxp
//! noiseextractor-cli mark in.flac --time 12.3:13.1 --freq 2000:4500 \
//!     --name "blackbird" --extract --project birds.nxp --save-project birds.nxp
//! noiseextractor-cli discover in.flac --sounds 6 --enable 2 --extract \
//!     --save-project birds.nxp
//! noiseextractor-cli process in.flac --project birds.nxp -o out.wav
//! noiseextractor-cli process in.flac --project birds.nxp --solo 2,3 -o birds.wav
//! noiseextractor-cli process in.flac --project birds.nxp -o birds.opus --bitrate 96
//! noiseextractor-cli spectrogram in.flac --project birds.nxp -o view.png
//! ```

// Every item, public or private, carries a doc comment (project rule).
#![warn(missing_docs, clippy::missing_docs_in_private_items)]

mod completions;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use completions::{CompletionShell, Program};
use noiseextractor_core::analysis::Selection;
use noiseextractor_core::discover::{discover, DiscoverParams, Discovery};
use noiseextractor_core::engine::{process_all, ChunkLayout, ChunkResult};
use noiseextractor_core::export::{create_writer, format_for_output, ExportOptions};
use noiseextractor_core::filter::{Filter, FilterAction, FilterKind, Project};
use noiseextractor_core::listen::{Listen, Pick};
use noiseextractor_core::noise::{mean_power, GainRule, NoiseEstimate};
use noiseextractor_core::normalize::{process_all_normalized, PEAK_DBFS_RANGE};
use noiseextractor_core::render::{render_rgb, View};
use noiseextractor_core::similar::find_in_range;
use noiseextractor_core::stft::Stft;
use noiseextractor_core::AudioData;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Command-line arguments.
#[derive(Parser)]
#[command(
    name = "noiseextractor-cli",
    version,
    about = "Suppress or extract noises in audio files using noiseextractor projects"
)]
struct Cli {
    #[command(subcommand)]
    /// The subcommand to run.
    command: Command,
    /// Worker threads (default: number of CPU cores).
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Output format: wav, flac, vorbis, mp3, opus or alac (default: from
    /// the output file's extension: .wav .flac .ogg .mp3 .opus .m4a).
    #[arg(long, global = true)]
    format: Option<String>,
    /// Bit rate in kbit/s for the lossy formats (Vorbis, MP3, Opus); the
    /// default is a transparent quality setting.
    #[arg(long, global = true)]
    bitrate: Option<u32>,
    /// STFT frames per processing chunk.
    #[arg(long, global = true, default_value_t = 256)]
    chunk_frames: usize,
}

/// Subcommands.
#[derive(Subcommand)]
enum Command {
    /// Apply all enabled filters of a project and write the result (WAV,
    /// FLAC, Ogg Vorbis, MP3, Opus or Apple Lossless, see --format).
    Process {
        /// Input audio file.
        input: PathBuf,
        /// Project file (default: background noise reduction only).
        #[arg(long)]
        project: Option<PathBuf>,
        /// Output audio file; its extension picks the format.
        #[arg(short, long)]
        output: PathBuf,
        /// Write only the sounds of filters N,M,... (1 = first in the
        /// project), as the GUI's "Hear the picked sound only" plays them.
        #[arg(long, value_name = "N,...", value_delimiter = ',', conflicts_with = "without")]
        solo: Vec<usize>,
        /// Write everything but the sounds of filters N,M,... (1 = first),
        /// as the GUI's "Hear everything but the picked sound" plays it.
        #[arg(long, value_name = "N,...", value_delimiter = ',')]
        without: Vec<usize>,
        /// Scale the output so its loudest sample reaches DBFS (default
        /// -1 when given without a value). One gain for the whole file, so
        /// the levels inside the recording keep their ratios.
        #[arg(long, value_name = "DBFS", num_args = 0..=1, default_missing_value = "-1", allow_hyphen_values = true)]
        normalize: Option<f32>,
    },
    /// Background noise reduction with explicit parameters.
    Denoise {
        /// Input audio file.
        input: PathBuf,
        /// Output audio file; its extension picks the format.
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
        /// Scale the output so its loudest sample reaches DBFS (default
        /// -1 when given without a value). One gain for the whole file, so
        /// the levels inside the recording keep their ratios.
        #[arg(long, value_name = "DBFS", num_args = 0..=1, default_missing_value = "-1", allow_hyphen_values = true)]
        normalize: Option<f32>,
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
    /// Find the sounds the recording is made of and add them to a project
    /// as suggested filters (switched off unless named with --enable).
    Discover {
        /// Input audio file.
        input: PathBuf,
        /// Number of spectral patterns to look for.
        #[arg(long, default_value_t = 6)]
        sounds: usize,
        /// Learn only from this time range "START:END" in seconds.
        #[arg(long)]
        time: Option<String>,
        /// Suggest every pattern on its own instead of grouping patterns
        /// that rise and fall together (and come from the same direction).
        #[arg(long)]
        no_grouping: bool,
        /// Suggestions to switch on and keep, by the number printed, e.g.
        /// "1,3".
        #[arg(long, value_delimiter = ',')]
        enable: Vec<usize>,
        /// Switched-on suggestions extract instead of suppress.
        #[arg(long)]
        extract: bool,
        /// Project to add the suggestions to (default: a new project).
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
        /// Output project file (by convention NAME.nxp).
        output: PathBuf,
    },
    /// Print a shell completion script, for this tool or for the GUI.
    ///
    /// For example `noiseextractor-cli completions bash >
    /// ~/.local/share/bash-completion/completions/noiseextractor-cli`.
    Completions {
        /// The shell to complete in.
        #[arg(value_enum)]
        shell: CompletionShell,
        /// The program to complete.
        #[arg(value_enum, default_value = "noiseextractor-cli")]
        program: Program,
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
    let threads = cli
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    anyhow::ensure!(cli.chunk_frames >= 1, "--chunk-frames must be at least 1");
    // Decided before the (possibly long) processing so that a typo in the
    // output name fails at once.
    let export = |output: &Path| -> Result<ExportOptions> {
        Ok(ExportOptions {
            format: format_for_output(output, cli.format.as_deref())?,
            bitrate_kbps: cli.bitrate,
        })
    };
    match cli.command {
        Command::Process {
            input,
            project,
            output,
            solo,
            without,
            normalize,
        } => {
            let options = export(&output)?;
            let audio = load(&input)?;
            let mut project = match project {
                Some(p) => Project::load(&p)?,
                None => Project::default(),
            };
            let (mode, numbers) = if !solo.is_empty() {
                (Listen::Solo, solo)
            } else {
                (Listen::Without, without)
            };
            if !numbers.is_empty() {
                let mut picks = Vec::new();
                for n in numbers {
                    anyhow::ensure!(
                        (1..=project.filters.len()).contains(&n),
                        "no filter {n}: the project has {} (numbered from 1)",
                        project.filters.len()
                    );
                    picks.push(Pick::Filter(n - 1));
                }
                project = project.for_listening(mode, &picks);
            }
            project.prepare(&audio);
            write_processed(&audio, &project, cli.chunk_frames, threads, &output, options, normalize)
        }
        Command::Denoise {
            input,
            output,
            reduction_db,
            rule,
            noise_from,
            save_project,
            normalize,
        } => {
            let options = output.as_deref().map(export).transpose()?;
            let audio = load(&input)?;
            let mut project = Project::default();
            let FilterKind::NoiseReduction(nr) = &mut project.filters[0].kind else {
                unreachable!("default project starts with noise reduction")
            };
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
                // Lets the GUI open the recording along with the project.
                project.audio = Some(input.clone());
                project.save(&p)?;
            }
            match (output, options) {
                (Some(o), Some(options)) => {
                    write_processed(&audio, &project, cli.chunk_frames, threads, &o, options, normalize)
                }
                _ => Ok(()),
            }
        }
        Command::Mark {
            input,
            time,
            freq,
            name,
            extract,
            project,
            save_project,
        } => {
            let audio = load(&input)?;
            let mut project = match project {
                Some(p) => Project::load(&p)?,
                None => Project::default(),
            };
            let (start_s, end_s) = parse_range(&time)?;
            let (low, high) = parse_range(&freq)?;
            let selection = Selection {
                start_s,
                end_s,
                low_hz: low as f32,
                high_hz: high as f32,
            };
            let action = if extract {
                FilterAction::Extract
            } else {
                FilterAction::Suppress
            };
            project.filters.push(Filter::event(name, selection, action));
            project.prepare(&audio);
            print_report(&project, &audio);
            if let Some(p) = save_project {
                // Lets the GUI open the recording along with the project.
                project.audio = Some(input.clone());
                project.save(&p)?;
            }
            Ok(())
        }
        Command::Discover {
            input,
            sounds,
            time,
            no_grouping,
            enable,
            extract,
            project,
            save_project,
        } => {
            let audio = load(&input)?;
            let mut project = match project {
                Some(p) => Project::load(&p)?,
                None => Project::default(),
            };
            let params = DiscoverParams {
                components: sounds,
                range_s: time.as_deref().map(parse_range).transpose()?,
                group: !no_grouping,
                ..DiscoverParams::default()
            };
            let found = discover(&audio, project.stft, &project.analysis_noise(), &params)?;
            print_discovery(&found);
            let first = project.add_discovery(found);
            let count = project.filters.len() - first;
            for n in enable {
                anyhow::ensure!((1..=count).contains(&n), "--enable {n}: there are {count} suggestions");
                let f = &mut project.filters[first + n - 1];
                f.enabled = true;
                f.suggested = false;
                if extract {
                    f.action = FilterAction::Extract;
                }
            }
            if let Some(p) = save_project {
                // Lets the GUI open the recording along with the project.
                project.audio = Some(input.clone());
                project.save(&p)?;
            }
            Ok(())
        }
        Command::Spectrogram {
            input,
            project,
            output,
            width,
            height,
            max_hz,
            log,
        } => {
            let audio = load(&input)?;
            let mut project = match project {
                Some(p) => Project::load(&p)?,
                None => Project {
                    filters: vec![],
                    ..Project::default()
                },
            };
            project.prepare(&audio);
            write_spectrogram(
                &audio,
                &project,
                cli.chunk_frames,
                threads,
                &output,
                width,
                height,
                max_hz,
                log,
            )
        }
        Command::NewProject { output } => Project::default().save(&output),
        Command::Completions { shell, program } => completions::write(shell, program, &mut std::io::stdout().lock()),
    }
}

/// Loads an audio file, refusing empty ones.
fn load(path: &Path) -> Result<AudioData> {
    let audio = AudioData::load(path)?;
    if audio.is_empty() {
        bail!("{} contains no samples", path.display());
    }
    eprintln!(
        "{}: {} ch, {} Hz, {:.1} s",
        path.display(),
        audio.channels.len(),
        audio.sample_rate,
        audio.duration()
    );
    Ok(audio)
}

/// Parses "A:B" into two numbers.
fn parse_range(s: &str) -> Result<(f64, f64)> {
    let (a, b) = s
        .split_once(':')
        .with_context(|| format!("expected START:END, got {s:?}"))?;
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

/// Processes everything and streams the result into `output`, scaled so
/// its peak reaches `normalize` dBFS if that is given.
fn write_processed(
    audio: &AudioData,
    project: &Project,
    chunk_frames: usize,
    threads: usize,
    output: &Path,
    options: ExportOptions,
    normalize: Option<f32>,
) -> Result<()> {
    if let Some(db) = normalize {
        let (lo, hi) = PEAK_DBFS_RANGE;
        anyhow::ensure!(
            (lo..=hi).contains(&db),
            "--normalize must be between {lo} and {hi} dBFS"
        );
    }
    let mut writer = create_writer(output, options, audio.sample_rate, audio.channels.len())?;
    let mut last_percent = usize::MAX;
    let done = process_all_normalized(
        audio,
        project,
        chunk_frames,
        threads,
        normalize,
        scratch_dir(output),
        |fraction| {
            let percent = (fraction * 100.0) as usize;
            if percent / 10 != last_percent / 10 {
                eprintln!("{percent:3}%");
                last_percent = percent;
            }
        },
        |block| writer.write(block),
    )?;
    writer.finish()?;
    if normalize.is_some() {
        eprintln!("normalized: gain {:+.1} dB", done.gain_db());
    }
    eprintln!("wrote {}", output.display());
    Ok(())
}

/// Directory for the scratch file of a normalized export: the output's,
/// which has room for the result anyway.
fn scratch_dir(output: &Path) -> &Path {
    match output.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    }
}

/// Prints what the analysis found for each event filter.
fn print_report(project: &Project, audio: &AudioData) {
    let sample_rate = audio.sample_rate;
    let mono = audio.mono();
    let stft = Stft::new(project.stft);
    let total_frames = project.stft.frame_count(mono.len()) as i64;
    for f in &project.filters {
        let FilterKind::Event(e) = &f.kind else { continue };
        let Some(a) = &e.analysis else { continue };
        let track: Vec<f32> = a
            .pitch_track
            .iter()
            .flatten()
            .map(|&b| project.stft.bin_to_hz(b, sample_rate))
            .collect();
        println!("filter {:?} ({:?})", f.name, f.action);
        if let Some((lo, hi)) = a.broadband {
            println!(
                "  broadband sound, {:.1} .. {:.1} kHz",
                project.stft.bin_to_hz(lo as f32, sample_rate) / 1000.0,
                project.stft.bin_to_hz(hi as f32, sample_rate) / 1000.0
            );
        } else if let (Some(lo), Some(hi)) = (
            track.iter().cloned().reduce(f32::min),
            track.iter().cloned().reduce(f32::max),
        ) {
            println!(
                "  dominant frequency in selection: {lo:.0} .. {hi:.0} Hz over {} frames",
                track.len()
            );
        } else {
            println!("  nothing above the noise floor in the selection");
        }
        for p in &a.partials {
            println!("  related partial at {:.3} x pitch (score {:.2})", p.ratio, p.score);
        }
        let comod = (0..a.bins)
            .filter(|&k| a.comodulated.iter().any(|f| f[k] > 0.5))
            .count();
        println!("  co-modulated bins: {comod}");
        if let Some(sig) = a.template.as_ref().and_then(|t| t.spatial.as_ref()) {
            for (i, p) in sig.pairs.iter().enumerate() {
                println!(
                    "  direction: channel {} vs 0: level {:+.1} dB, delay {:+.2} samples",
                    i + 1,
                    p.level_db,
                    p.delay_samples
                );
            }
        }
        match (&a.template, e.use_similar) {
            (Some(tpl), true) => {
                if let Some(r) = &tpl.rhythm {
                    let hz = sample_rate as f32 / (project.stft.hop as f32 * r.period);
                    println!("  pulsed at {hz:.1} Hz: searching by rhythm and band");
                }
                let found = find_in_range(tpl, &mono, &audio.channels, &stft, 0..total_frames, &e.similar);
                println!("  same sound found {} times:", found.len());
                for o in found {
                    let t = o.frame as f64 * project.stft.hop as f64 / sample_rate as f64;
                    let note = if o.is_original(tpl) { ", the marked one" } else { "" };
                    let place = o
                        .spatial_agreement
                        .map(|a| format!(", same place {a:.2}"))
                        .unwrap_or_default();
                    let len = if tpl.rhythm.is_some() {
                        let s = o.frames as f64 * project.stft.hop as f64 / sample_rate as f64;
                        format!(", {s:.2} s long")
                    } else {
                        String::new()
                    };
                    println!(
                        "    at {t:.3} s (match {:.2}, shifted {} bins{len}{place}{note})",
                        o.score, o.shift
                    );
                }
            }
            (None, _) => println!("  nothing distinctive to search for elsewhere"),
            (_, false) => {}
        }
    }
}

/// Prints the suggestions of an automatic search, numbered as `--enable`
/// expects them.
fn print_discovery(found: &Discovery) {
    let d = &found.decomposition;
    println!(
        "learned {} patterns from {} frames",
        d.patterns.len(),
        found.frames_used
    );
    for (n, group) in found.groups.iter().enumerate() {
        let share: f32 = group.iter().map(|&c| found.energy[c]).sum();
        println!("suggestion {} ({:.0} % of the sound):", n + 1, share * 100.0);
        for &c in group {
            let direction = d.directions[c]
                .as_ref()
                .and_then(|s| s.pairs.first())
                .map(|p| format!(", level {:+.1} dB, delay {:+.2} samples", p.level_db, p.delay_samples))
                .unwrap_or_default();
            println!("  pattern {}: {}{direction}", c + 1, d.labels[c]);
        }
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
        let slim = ChunkResult {
            audio: Vec::new(),
            ..clone_layers(r)
        };
        chunks.insert(r.index, Arc::new(slim));
        Ok(())
    })?;
    let layout = ChunkLayout {
        hop: project.stft.hop,
        chunk_frames,
        total_samples: audio.len(),
    };
    let view = View {
        start_sample: 0.0,
        samples_per_px: audio.len() as f64 / width as f64,
        low_hz: if log { 50.0 } else { 0.0 },
        high_hz: max_hz.unwrap_or(audio.sample_rate as f32 / 2.0),
        log_freq: log,
        db_min: -100.0,
        db_max: -10.0,
    };
    let rgb = render_rgb(
        &view,
        width,
        height,
        layout,
        audio.sample_rate,
        project.stft.fft_size,
        &|c| chunks.get(&c).cloned(),
    );
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
        found: r.found.clone(),
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
