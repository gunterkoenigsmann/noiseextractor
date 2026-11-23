//! # noiseextractor-python
//!
//! The native part of the Python package `noiseextractor`: the extension
//! module `noiseextractor._native`, built by maturin (`pyproject.toml`).
//!
//! It is deliberately thin. Every function takes and returns plain Python
//! values (strings, numbers, lists, dicts, bytes); the typed, documented
//! API that users see (`denoise()`, `process()`, `mark()`, `discover()`,
//! the `*_array()` variants and their result classes) is the pure Python
//! layer in `python/noiseextractor/`. The work itself is
//! `noiseextractor-core`'s, reached through the building blocks of the
//! command line tool (`noiseextractor-cli`'s library), so the package, the
//! command line tool and the GUI process sound identically.
//!
//! Times are seconds (`float`), frequencies Hz. Sample buffers cross the
//! boundary as native-endian 32-bit floats in `bytes`, one buffer per
//! channel, so neither side needs NumPy.
//!
//! The work runs with the interpreter lock released (`Python::detach`), so
//! other Python threads keep running during a long run.

// Every item, public or private, carries a doc comment (project rule).
#![warn(missing_docs, clippy::missing_docs_in_private_items)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use noiseextractor_cli::report::{self, EventReport, Suggestion};
use noiseextractor_cli::{default_threads, learn_noise, load_audio, write_processed, write_spectrogram};
use noiseextractor_core::analysis::Selection;
use noiseextractor_core::discover::{discover as find_sounds, DiscoverParams};
use noiseextractor_core::export::{format_for_output, ExportOptions};
use noiseextractor_core::filter::{Filter, FilterAction, FilterKind, Project};
use noiseextractor_core::listen::{Listen, Pick};
use noiseextractor_core::noise::{GainRule, NoiseEstimate};
use noiseextractor_core::normalize::{process_all_normalized, PEAK_DBFS_RANGE};
use noiseextractor_core::AudioData;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

pyo3::create_exception!(
    _native,
    Error,
    PyException,
    "A file could not be read, processed or written (noiseextractor.Error)."
);

/// STFT frames per processing chunk; the command line tool's default.
const CHUNK_FRAMES: usize = 256;

/// What can go wrong in a call, before it becomes a Python exception.
enum CallError {
    /// A bad argument: unknown rule, filter number out of range. Becomes a
    /// `ValueError`.
    Value(String),
    /// Reading, processing or writing failed. Becomes a
    /// `noiseextractor.Error`.
    File(String),
}

impl From<CallError> for PyErr {
    fn from(e: CallError) -> PyErr {
        match e {
            CallError::Value(m) => PyValueError::new_err(m),
            CallError::File(m) => Error::new_err(m),
        }
    }
}

impl From<anyhow::Error> for CallError {
    fn from(e: anyhow::Error) -> Self {
        // `{:#}` prints the whole chain in one line, `outer: inner`, the
        // way the command line tool reports errors.
        CallError::File(format!("{e:#}"))
    }
}

/// Shorthand for the results of the functions below.
type CallResult<T> = Result<T, CallError>;

/// Fails with a `ValueError` unless `ok`.
fn check(ok: bool, message: impl FnOnce() -> String) -> CallResult<()> {
    if ok {
        Ok(())
    } else {
        Err(CallError::Value(message()))
    }
}

/// The worker threads to use: `threads`, or one per CPU core.
fn threads_or_default(threads: Option<usize>) -> CallResult<usize> {
    match threads {
        Some(0) => Err(CallError::Value("threads must be at least 1".into())),
        Some(n) => Ok(n),
        None => Ok(default_threads()),
    }
}

/// Loads a project file, or the default project (background noise
/// reduction only) without one.
fn load_project(path: Option<&Path>) -> CallResult<Project> {
    Ok(match path {
        Some(p) => Project::load(p)?,
        None => Project::default(),
    })
}

/// Saves `project` to `path`, naming `recording` in it so that the GUI
/// opens the recording along with the project.
fn save_project(mut project: Project, path: &Path, recording: Option<&Path>) -> CallResult<()> {
    project.audio = recording.map(Path::to_path_buf);
    Ok(project.save(path)?)
}

/// Turns `project` into the project that keeps only the sounds of filters
/// `solo` or everything but those of `without` (numbered from 1, as the
/// command line's `--solo` and `--without`); unchanged if both are empty.
fn apply_listening(project: Project, solo: &[usize], without: &[usize]) -> CallResult<Project> {
    check(solo.is_empty() || without.is_empty(), || {
        "give solo or without, not both".into()
    })?;
    let (mode, numbers) = if solo.is_empty() {
        (Listen::Without, without)
    } else {
        (Listen::Solo, solo)
    };
    if numbers.is_empty() {
        return Ok(project);
    }
    let mut picks = Vec::new();
    for &n in numbers {
        check((1..=project.filters.len()).contains(&n), || {
            format!(
                "no filter {n}: the project has {} (numbered from 1)",
                project.filters.len()
            )
        })?;
        picks.push(Pick::Filter(n - 1));
    }
    Ok(project.for_listening(mode, &picks))
}

/// The parameters of `denoise`, shared by the file and the sample variant.
struct DenoiseParams {
    /// Maximum attenuation in dB.
    reduction_db: f32,
    /// `"lsa"` or `"wiener"`.
    rule: String,
    /// Noise-only stretch to learn the noise from, in seconds.
    noise_from: Option<(f64, f64)>,
}

/// The project `denoise` runs: the default project's noise reduction with
/// the parameters `p`.
fn denoise_project(audio: &AudioData, p: &DenoiseParams) -> CallResult<Project> {
    check(p.reduction_db.is_finite() && p.reduction_db >= 0.0, || {
        format!("reduction_db must be 0 or more, not {}", p.reduction_db)
    })?;
    let rule = match p.rule.to_ascii_lowercase().as_str() {
        "lsa" | "mmse-lsa" | "mmse_lsa" => GainRule::MmseLsa,
        "wiener" => GainRule::Wiener,
        other => {
            return Err(CallError::Value(format!(
                "unknown rule {other:?}: use \"lsa\" or \"wiener\""
            )))
        }
    };
    let mut project = Project::default();
    let FilterKind::NoiseReduction(nr) = &mut project.filters[0].kind else {
        unreachable!("the default project starts with noise reduction")
    };
    nr.reduction_db = p.reduction_db;
    nr.rule = rule;
    if let Some((a, b)) = p.noise_from {
        let (a, b) = (a.min(b), a.max(b));
        nr.estimate = NoiseEstimate::Profile(learn_noise(audio, &project.stft, a, b)?);
    }
    Ok(project)
}

/// Checks a normalisation target in dBFS.
fn check_normalize(normalize: Option<f32>) -> CallResult<()> {
    if let Some(db) = normalize {
        let (lo, hi) = PEAK_DBFS_RANGE;
        check((lo..=hi).contains(&db), || {
            format!("normalize must be between {lo} and {hi} dBFS, not {db}")
        })?;
    }
    Ok(())
}

/// How to write an output file: from `format` or the file's extension.
fn export_options(output: &Path, format: Option<&str>, bitrate: Option<u32>) -> CallResult<ExportOptions> {
    let format = format_for_output(output, format).map_err(|e| CallError::Value(format!("{e:#}")))?;
    Ok(ExportOptions {
        format,
        bitrate_kbps: bitrate,
    })
}

/// Processes `audio` with `project` and writes `output`; returns
/// `{"peak": ..., "gain_db": ...}` (gain 0 dB unless normalised).
fn write(
    py: Python<'_>,
    audio: &AudioData,
    project: &Project,
    output: &Path,
    options: ExportOptions,
    normalize: Option<f32>,
    threads: usize,
) -> PyResult<Py<PyDict>> {
    let done = py
        .detach(|| {
            write_processed(
                audio,
                project,
                CHUNK_FRAMES,
                threads,
                output,
                options,
                normalize,
                |_| {},
            )
        })
        .map_err(CallError::from)?;
    let d = PyDict::new(py);
    d.set_item("peak", done.peak)?;
    d.set_item("gain_db", done.gain_db())?;
    Ok(d.unbind())
}

/// Background noise reduction of a file: the GUI's default filter with
/// explicit parameters. Writes `output` if given, saves the project to
/// `save_project` if given. Returns `{"peak", "gain_db"}` or `None`
/// without an output.
#[pyfunction]
#[pyo3(signature = (input, output, reduction_db, rule, noise_from, normalize, format, bitrate, save_project, threads))]
#[allow(clippy::too_many_arguments)]
fn denoise(
    py: Python<'_>,
    input: PathBuf,
    output: Option<PathBuf>,
    reduction_db: f32,
    rule: String,
    noise_from: Option<(f64, f64)>,
    normalize: Option<f32>,
    format: Option<String>,
    bitrate: Option<u32>,
    save_project: Option<PathBuf>,
    threads: Option<usize>,
) -> PyResult<Option<Py<PyDict>>> {
    let threads = threads_or_default(threads)?;
    check_normalize(normalize)?;
    // Decided before the (possibly long) processing, so that a typo in the
    // output name fails at once.
    let options = output
        .as_deref()
        .map(|o| export_options(o, format.as_deref(), bitrate))
        .transpose()?;
    let params = DenoiseParams {
        reduction_db,
        rule,
        noise_from,
    };
    let (audio, project) = py.detach(|| -> CallResult<_> {
        let audio = load_audio(&input)?;
        let project = denoise_project(&audio, &params)?;
        Ok((audio, project))
    })?;
    if let Some(p) = &save_project {
        self::save_project(project.clone(), p, Some(&input))?;
    }
    match (output, options) {
        (Some(o), Some(options)) => Ok(Some(write(py, &audio, &project, &o, options, normalize, threads)?)),
        _ => Ok(None),
    }
}

/// Applies a project's enabled filters to a file and writes the result.
/// `solo`/`without` keep only, or drop, the sounds of the filters with
/// these numbers (from 1). Returns `{"peak", "gain_db"}`.
#[pyfunction]
#[pyo3(signature = (input, output, project, solo, without, normalize, format, bitrate, threads))]
#[allow(clippy::too_many_arguments)]
fn process(
    py: Python<'_>,
    input: PathBuf,
    output: PathBuf,
    project: Option<PathBuf>,
    solo: Vec<usize>,
    without: Vec<usize>,
    normalize: Option<f32>,
    format: Option<String>,
    bitrate: Option<u32>,
    threads: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let threads = threads_or_default(threads)?;
    check_normalize(normalize)?;
    let options = export_options(&output, format.as_deref(), bitrate)?;
    let project = apply_listening(load_project(project.as_deref())?, &solo, &without)?;
    let (audio, project) = py.detach(|| -> CallResult<_> {
        let audio = load_audio(&input)?;
        let mut project = project;
        project.prepare(&audio);
        Ok((audio, project))
    })?;
    write(py, &audio, &project, &output, options, normalize, threads)
}

/// An event report as a dict (the fields of
/// [`noiseextractor_cli::report::EventReport`]).
fn report_dict<'py>(py: Python<'py>, r: &EventReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("name", &r.name)?;
    d.set_item("extract", r.action == FilterAction::Extract)?;
    d.set_item("broadband_hz", r.broadband_hz)?;
    d.set_item("pitch_hz", r.pitch_hz)?;
    d.set_item("pitch_frames", r.pitch_frames)?;
    let partials = PyList::empty(py);
    for p in &r.partials {
        partials.append((p.ratio, p.score))?;
    }
    d.set_item("partials", partials)?;
    d.set_item("comodulated_bins", r.comodulated_bins)?;
    let directions = PyList::empty(py);
    for x in &r.directions {
        directions.append((x.channel, x.level_db, x.delay_samples))?;
    }
    d.set_item("directions", directions)?;
    d.set_item("pulse_hz", r.pulse_hz)?;
    d.set_item("searchable", r.searchable)?;
    match &r.occurrences {
        None => d.set_item("occurrences", py.None())?,
        Some(found) => {
            let list = PyList::empty(py);
            for o in found {
                let od = PyDict::new(py);
                od.set_item("time_s", o.time_s)?;
                od.set_item("score", o.score)?;
                od.set_item("shift_bins", o.shift_bins)?;
                od.set_item("length_s", o.length_s)?;
                od.set_item("same_place", o.same_place)?;
                od.set_item("marked", o.marked)?;
                list.append(od)?;
            }
            d.set_item("occurrences", list)?;
        }
    }
    d.set_item("text", r.to_string())?;
    Ok(d)
}

/// Marks a time and frequency rectangle as a sound, analyses what belongs
/// to it and where else it occurs, and adds it as a filter to `project`
/// (or a new project). Returns the report dict; saves the extended project
/// to `save_project` if given.
#[pyfunction]
#[pyo3(signature = (input, time, freq, name, extract, project, save_project))]
#[allow(clippy::too_many_arguments)]
fn mark(
    py: Python<'_>,
    input: PathBuf,
    time: (f64, f64),
    freq: (f32, f32),
    name: String,
    extract: bool,
    project: Option<PathBuf>,
    save_project: Option<PathBuf>,
) -> PyResult<Py<PyDict>> {
    let mut project = load_project(project.as_deref())?;
    let selection = Selection::rect(time.0, time.1, freq.0, freq.1);
    check(
        selection.end_s > selection.start_s && selection.high_hz > selection.low_hz,
        || "the marked time and frequency ranges must not be empty".into(),
    )?;
    let action = if extract {
        FilterAction::Extract
    } else {
        FilterAction::Suppress
    };
    project.filters.push(Filter::event(name, selection, action));
    let (project, found) = py.detach(|| -> CallResult<_> {
        let audio = load_audio(&input)?;
        let mut project = project;
        project.prepare(&audio);
        let found = report::event_report(&project, project.filters.len() - 1, &audio);
        Ok((project, found))
    })?;
    if let Some(p) = &save_project {
        self::save_project(project, p, Some(&input))?;
    }
    let found = found.ok_or_else(|| CallError::File("the marked sound could not be analysed".into()))?;
    Ok(report_dict(py, &found)?.unbind())
}

/// A suggestion as a dict (the fields of
/// [`noiseextractor_cli::report::Suggestion`]).
fn suggestion_dict<'py>(py: Python<'py>, s: &Suggestion) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("number", s.number)?;
    d.set_item("share", s.share)?;
    let patterns = PyList::empty(py);
    for p in &s.patterns {
        let pd = PyDict::new(py);
        pd.set_item("number", p.number)?;
        pd.set_item("label", &p.label)?;
        pd.set_item("direction", p.direction)?;
        patterns.append(pd)?;
    }
    d.set_item("patterns", patterns)?;
    d.set_item("text", s.to_string())?;
    Ok(d)
}

/// Finds the sounds a recording is made of (non-negative matrix
/// factorisation) and adds them to `project` (or a new project) as
/// suggested filters, switched off except those numbered in `enable`.
/// Returns the list of suggestion dicts; saves the project if asked to.
#[pyfunction]
#[pyo3(signature = (input, sounds, time, grouping, enable, extract, project, save_project))]
#[allow(clippy::too_many_arguments)]
fn discover(
    py: Python<'_>,
    input: PathBuf,
    sounds: usize,
    time: Option<(f64, f64)>,
    grouping: bool,
    enable: Vec<usize>,
    extract: bool,
    project: Option<PathBuf>,
    save_project: Option<PathBuf>,
) -> PyResult<Py<PyList>> {
    check(sounds >= 1, || "sounds must be at least 1".into())?;
    let mut project = load_project(project.as_deref())?;
    let params = DiscoverParams {
        components: sounds,
        range_s: time.map(|(a, b)| (a.min(b), a.max(b))),
        group: grouping,
        ..DiscoverParams::default()
    };
    let found = py.detach(|| -> CallResult<_> {
        let audio = load_audio(&input)?;
        Ok(find_sounds(&audio, project.stft, &project.analysis_noise(), &params)?)
    })?;
    let suggestions = report::suggestions(&found);
    let first = project.add_discovery(found);
    let count = project.filters.len() - first;
    for n in enable {
        check((1..=count).contains(&n), || {
            format!("enable {n}: there are {count} suggestions")
        })?;
        let f = &mut project.filters[first + n - 1];
        f.enabled = true;
        f.suggested = false;
        if extract {
            f.action = FilterAction::Extract;
        }
    }
    if let Some(p) = &save_project {
        self::save_project(project, p, Some(&input))?;
    }
    let list = PyList::empty(py);
    for s in &suggestions {
        list.append(suggestion_dict(py, s)?)?;
    }
    Ok(list.unbind())
}

/// Renders the processed spectrogram with the GUI's colours into a PNG.
#[pyfunction]
#[pyo3(signature = (input, output, project, width, height, max_hz, log, threads))]
#[allow(clippy::too_many_arguments)]
fn spectrogram(
    py: Python<'_>,
    input: PathBuf,
    output: PathBuf,
    project: Option<PathBuf>,
    width: u32,
    height: u32,
    max_hz: Option<f32>,
    log: bool,
    threads: Option<usize>,
) -> PyResult<()> {
    let threads = threads_or_default(threads)?;
    check(width > 0 && height > 0, || "the image must not be empty".into())?;
    let mut project = match project {
        Some(p) => Project::load(&p).map_err(CallError::from)?,
        // Without a project: the recording as it is, no filters.
        None => Project {
            filters: vec![],
            ..Project::default()
        },
    };
    py.detach(|| -> CallResult<()> {
        let audio = load_audio(&input)?;
        project.prepare(&audio);
        Ok(write_spectrogram(
            &audio,
            &project,
            CHUNK_FRAMES,
            threads,
            &output,
            width,
            height,
            max_hz,
            log,
        )?)
    })?;
    Ok(())
}

/// Writes a project with the default filters (background noise reduction).
#[pyfunction]
fn new_project(path: PathBuf) -> PyResult<()> {
    Ok(Project::default().save(&path).map_err(CallError::from)?)
}

/// Native-endian 32-bit floats from `bytes`, as the Python layer packs
/// them (`array('f')` or a float32 NumPy array).
fn floats(data: &[u8]) -> CallResult<Vec<f32>> {
    check(data.len().is_multiple_of(4), || {
        "sample buffers hold 4 bytes per sample".into()
    })?;
    // The length was checked above, so no bytes are left over.
    Ok(data.as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes(*b)).collect())
}

/// Processes in-memory samples with `project` and returns the channels as
/// native float32 `bytes`.
fn run_on_samples(py: Python<'_>, audio: &AudioData, project: &Project, threads: usize) -> PyResult<Py<PyList>> {
    let out = py
        .detach(|| -> CallResult<Vec<Vec<f32>>> {
            let mut out: Vec<Vec<f32>> = vec![Vec::with_capacity(audio.len()); audio.channels.len()];
            process_all_normalized(
                audio,
                project,
                CHUNK_FRAMES,
                threads,
                None,
                // Only a normalised run writes a scratch file.
                &noiseextractor_core::samples::scratch_dir(),
                |_| {},
                |block| {
                    for (o, b) in out.iter_mut().zip(block) {
                        o.extend_from_slice(b);
                    }
                    Ok(())
                },
            )?;
            Ok(out)
        })
        .map_err(PyErr::from)?;
    let list = PyList::empty(py);
    for c in out {
        let bytes: Vec<u8> = c.iter().flat_map(|v| v.to_ne_bytes()).collect();
        list.append(PyBytes::new(py, &bytes))?;
    }
    Ok(list.unbind())
}

/// Builds a recording from per-channel float32 buffers.
fn audio_from(channels: Vec<Vec<u8>>, sample_rate: u32) -> CallResult<AudioData> {
    check(!channels.is_empty(), || "no channels given".into())?;
    check(sample_rate > 0, || "sample_rate must be positive".into())?;
    let channels = channels.iter().map(|c| floats(c)).collect::<CallResult<Vec<_>>>()?;
    let len = channels[0].len();
    check(len > 0, || "the samples are empty".into())?;
    check(channels.iter().all(|c| c.len() == len), || {
        "all channels must have the same number of samples".into()
    })?;
    Ok(AudioData::new(sample_rate, channels))
}

/// `denoise` on samples in memory: one float32 `bytes` per channel in and
/// out.
#[pyfunction]
#[pyo3(signature = (channels, sample_rate, reduction_db, rule, noise_from, threads))]
fn denoise_samples(
    py: Python<'_>,
    channels: Vec<Vec<u8>>,
    sample_rate: u32,
    reduction_db: f32,
    rule: String,
    noise_from: Option<(f64, f64)>,
    threads: Option<usize>,
) -> PyResult<Py<PyList>> {
    let threads = threads_or_default(threads)?;
    let audio = audio_from(channels, sample_rate)?;
    let params = DenoiseParams {
        reduction_db,
        rule,
        noise_from,
    };
    let project = denoise_project(&audio, &params)?;
    run_on_samples(py, &audio, &project, threads)
}

/// `process` on samples in memory: one float32 `bytes` per channel in and
/// out. The project comes from the file `project`.
#[pyfunction]
#[pyo3(signature = (channels, sample_rate, project, solo, without, threads))]
fn process_samples(
    py: Python<'_>,
    channels: Vec<Vec<u8>>,
    sample_rate: u32,
    project: Option<PathBuf>,
    solo: Vec<usize>,
    without: Vec<usize>,
    threads: Option<usize>,
) -> PyResult<Py<PyList>> {
    let threads = threads_or_default(threads)?;
    let audio = audio_from(channels, sample_rate)?;
    let mut project = apply_listening(load_project(project.as_deref())?, &solo, &without)?;
    py.detach(|| project.prepare(&audio));
    run_on_samples(py, &audio, &project, threads)
}

/// Decodes an audio file (WAV, FLAC, Ogg Vorbis, MP3, AAC/M4A, ALAC, ...)
/// into one float32 `bytes` per channel; returns `(channels, sample_rate)`.
#[pyfunction]
fn load(py: Python<'_>, path: PathBuf) -> PyResult<(Py<PyList>, u32)> {
    let audio = py.detach(|| load_audio(&path)).map_err(CallError::from)?;
    let list = PyList::empty(py);
    for c in &audio.channels {
        let bytes: Vec<u8> = c.iter().flat_map(|v| v.to_ne_bytes()).collect();
        list.append(PyBytes::new(py, &bytes))?;
    }
    Ok((list.unbind(), audio.sample_rate))
}

/// Runs the command line tool `noiseextractor-cli` in this process with
/// `argv` (program name first) and returns its exit status.
#[pyfunction]
fn run_cli(py: Python<'_>, argv: Vec<OsString>) -> i32 {
    py.detach(|| noiseextractor_cli::run(argv))
}

/// The extension module `noiseextractor._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("Error", m.py().get_type::<Error>())?;
    m.add_function(wrap_pyfunction!(denoise, m)?)?;
    m.add_function(wrap_pyfunction!(process, m)?)?;
    m.add_function(wrap_pyfunction!(mark, m)?)?;
    m.add_function(wrap_pyfunction!(discover, m)?)?;
    m.add_function(wrap_pyfunction!(spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(new_project, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_samples, m)?)?;
    m.add_function(wrap_pyfunction!(process_samples, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    Ok(())
}
