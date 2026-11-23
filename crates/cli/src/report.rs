//! What the analysis of a marked sound and an automatic search found, as
//! data (for the Python package) and as the text the `mark` and `discover`
//! subcommands print.
//!
//! The data structures hold plain numbers in user units (seconds, Hz,
//! dB), so front ends need not know the STFT grid.

use std::fmt;

use noiseextractor_core::discover::Discovery;
use noiseextractor_core::filter::{FilterAction, FilterKind, Project};
use noiseextractor_core::similar::find_in_range;
use noiseextractor_core::stft::Stft;
use noiseextractor_core::AudioData;

/// Where the marked sound comes from, relative to the first channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Direction {
    /// The other channel (1 = the second one).
    pub channel: usize,
    /// Level of that channel relative to channel 0 in dB.
    pub level_db: f32,
    /// Delay of that channel relative to channel 0 in samples.
    pub delay_samples: f32,
}

/// A related partial of the marked sound: an overtone or a resonance.
#[derive(Clone, Debug, PartialEq)]
pub struct Partial {
    /// Frequency as a multiple of the sound's pitch (2.0 = octave).
    pub ratio: f32,
    /// How sure the analysis is that it belongs to the sound (0..1).
    pub score: f32,
}

/// One place where the marked sound occurs (again).
#[derive(Clone, Debug, PartialEq)]
pub struct Occurrence {
    /// Start in seconds.
    pub time_s: f64,
    /// Shape correlation with the marked sound (0..1), lowered when the
    /// direction disagrees.
    pub score: f32,
    /// Frequency shift relative to the marked sound, in STFT bins.
    pub shift_bins: i32,
    /// Length in seconds for pulsed sounds (found burst by burst), `None`
    /// for sounds matched by their shape (as long as the marked part).
    pub length_s: Option<f64>,
    /// Agreement of the direction with the marked sound's (1 = same
    /// place), `None` for mono recordings.
    pub same_place: Option<f32>,
    /// Whether this is the marked sound itself.
    pub marked: bool,
}

/// What the analysis found for one marked sound (an event filter).
#[derive(Clone, Debug, PartialEq)]
pub struct EventReport {
    /// The filter's name.
    pub name: String,
    /// Whether the filter suppresses or extracts its sound.
    pub action: FilterAction,
    /// For a noise-like sound without pitch: its band in Hz.
    pub broadband_hz: Option<(f32, f32)>,
    /// For a pitched sound: lowest and highest dominant frequency in Hz.
    pub pitch_hz: Option<(f32, f32)>,
    /// Number of STFT frames the pitch was tracked over.
    pub pitch_frames: usize,
    /// Partials found to belong to the sound.
    pub partials: Vec<Partial>,
    /// Number of frequency bins whose level rises and falls with the sound.
    pub comodulated_bins: usize,
    /// Stereo (or multi-channel) cues, empty for mono recordings.
    pub directions: Vec<Direction>,
    /// For pulsed sounds: the pulse rate in Hz.
    pub pulse_hz: Option<f32>,
    /// Whether the sound had anything distinctive to search for elsewhere.
    pub searchable: bool,
    /// Where the same sound occurs, if the filter searches for it.
    pub occurrences: Option<Vec<Occurrence>>,
}

/// Analyses every event filter of a prepared project (see
/// [`Project::prepare`]) and searches the whole recording for repetitions
/// of the marked sounds.
pub fn event_reports(project: &Project, audio: &AudioData) -> Vec<EventReport> {
    (0..project.filters.len())
        .filter_map(|i| event_report(project, i, audio))
        .collect()
}

/// Analyses the filter number `index` (from 0) of a prepared project like
/// [`event_reports`]; `None` if it is no analysed event filter.
pub fn event_report(project: &Project, index: usize, audio: &AudioData) -> Option<EventReport> {
    let sample_rate = audio.sample_rate;
    let cfg = project.stft;
    let mono = audio.mono();
    let seconds = |frames: f64| frames * cfg.hop as f64 / sample_rate as f64;
    let f = project.filters.get(index)?;
    let FilterKind::Event(e) = &f.kind else { return None };
    let a = e.analysis.as_ref()?;
    let track: Vec<f32> = a
        .pitch_track
        .iter()
        .flatten()
        .map(|&b| cfg.bin_to_hz(b, sample_rate))
        .collect();
    let pitch_hz = match (
        track.iter().cloned().reduce(f32::min),
        track.iter().cloned().reduce(f32::max),
    ) {
        (Some(lo), Some(hi)) if a.broadband.is_none() => Some((lo, hi)),
        _ => None,
    };
    let directions = a
        .template
        .as_ref()
        .and_then(|t| t.spatial.as_ref())
        .map(|sig| {
            sig.pairs
                .iter()
                .enumerate()
                .map(|(i, p)| Direction {
                    channel: i + 1,
                    level_db: p.level_db,
                    delay_samples: p.delay_samples,
                })
                .collect()
        })
        .unwrap_or_default();
    let pulse_hz = a
        .template
        .as_ref()
        .and_then(|t| t.rhythm.as_ref())
        .map(|r| sample_rate as f32 / (cfg.hop as f32 * r.period));
    let occurrences = match (&a.template, e.use_similar) {
        (Some(tpl), true) => Some(
            find_in_range(
                tpl,
                mono,
                &audio.channels,
                &Stft::new(cfg),
                0..cfg.frame_count(mono.len()) as i64,
                &e.similar,
            )
            .into_iter()
            .map(|o| Occurrence {
                time_s: seconds(o.frame as f64),
                score: o.score,
                shift_bins: o.shift,
                length_s: tpl.rhythm.as_ref().map(|_| seconds(o.frames as f64)),
                same_place: o.spatial_agreement,
                marked: o.is_original(tpl),
            })
            .collect(),
        ),
        _ => None,
    };
    Some(EventReport {
        name: f.name.clone(),
        action: f.action,
        broadband_hz: a.broadband.map(|(lo, hi)| {
            (
                cfg.bin_to_hz(lo as f32, sample_rate),
                cfg.bin_to_hz(hi as f32, sample_rate),
            )
        }),
        pitch_hz,
        pitch_frames: track.len(),
        partials: a
            .partials
            .iter()
            .map(|p| Partial {
                ratio: p.ratio,
                score: p.score,
            })
            .collect(),
        // A bin counts when it follows the sound in any frame.
        comodulated_bins: (0..a.bins)
            .filter(|&k| a.comodulated.iter().any(|f| f[k] > 0.5))
            .count(),
        directions,
        pulse_hz,
        searchable: a.template.is_some(),
        occurrences,
    })
}

/// The text the `mark` subcommand prints for one marked sound.
impl fmt::Display for EventReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "filter {:?} ({:?})", self.name, self.action)?;
        if let Some((lo, hi)) = self.broadband_hz {
            writeln!(f, "  broadband sound, {:.1} .. {:.1} kHz", lo / 1000.0, hi / 1000.0)?;
        } else if let Some((lo, hi)) = self.pitch_hz {
            writeln!(
                f,
                "  dominant frequency in selection: {lo:.0} .. {hi:.0} Hz over {} frames",
                self.pitch_frames
            )?;
        } else {
            writeln!(f, "  nothing above the noise floor in the selection")?;
        }
        for p in &self.partials {
            writeln!(f, "  related partial at {:.3} x pitch (score {:.2})", p.ratio, p.score)?;
        }
        writeln!(f, "  co-modulated bins: {}", self.comodulated_bins)?;
        for d in &self.directions {
            writeln!(
                f,
                "  direction: channel {} vs 0: level {:+.1} dB, delay {:+.2} samples",
                d.channel, d.level_db, d.delay_samples
            )?;
        }
        if !self.searchable {
            return writeln!(f, "  nothing distinctive to search for elsewhere");
        }
        let Some(found) = &self.occurrences else {
            return Ok(());
        };
        if let Some(hz) = self.pulse_hz {
            writeln!(f, "  pulsed at {hz:.1} Hz: searching by rhythm and band")?;
        }
        writeln!(f, "  same sound found {} times:", found.len())?;
        for o in found {
            let note = if o.marked { ", the marked one" } else { "" };
            let place = o.same_place.map(|a| format!(", same place {a:.2}")).unwrap_or_default();
            let len = o.length_s.map(|s| format!(", {s:.2} s long")).unwrap_or_default();
            writeln!(
                f,
                "    at {:.3} s (match {:.2}, shifted {} bins{len}{place}{note})",
                o.time_s, o.score, o.shift_bins
            )?;
        }
        Ok(())
    }
}

/// One spectral pattern of an automatic search.
#[derive(Clone, Debug, PartialEq)]
pub struct Pattern {
    /// The pattern's number (1 = first learned).
    pub number: usize,
    /// A short description such as "tonal, peak 660 Hz".
    pub label: String,
    /// Stereo cue of the pattern relative to channel 0, if any:
    /// `(level in dB, delay in samples)` of the second channel.
    pub direction: Option<(f32, f32)>,
}

/// One suggested filter of an automatic search: patterns that rise and
/// fall together.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    /// The number `--enable` (or `enable=` in Python) expects, from 1.
    pub number: usize,
    /// Share of the learned frames' magnitude it explains (0..1).
    pub share: f32,
    /// Its patterns.
    pub patterns: Vec<Pattern>,
}

/// The suggestions of an automatic search, strongest first, numbered as
/// [`Project::add_discovery`] adds them.
pub fn suggestions(found: &Discovery) -> Vec<Suggestion> {
    let d = &found.decomposition;
    found
        .groups
        .iter()
        .enumerate()
        .map(|(n, group)| Suggestion {
            number: n + 1,
            share: group.iter().map(|&c| found.energy[c]).sum(),
            patterns: group
                .iter()
                .map(|&c| Pattern {
                    number: c + 1,
                    label: d.labels[c].clone(),
                    direction: d.directions[c]
                        .as_ref()
                        .and_then(|s| s.pairs.first())
                        .map(|p| (p.level_db, p.delay_samples)),
                })
                .collect(),
        })
        .collect()
}

/// The text the `discover` subcommand prints for one suggestion.
impl fmt::Display for Suggestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "suggestion {} ({:.0} % of the sound):",
            self.number,
            self.share * 100.0
        )?;
        for p in &self.patterns {
            let direction = p
                .direction
                .map(|(level, delay)| format!(", level {level:+.1} dB, delay {delay:+.2} samples"))
                .unwrap_or_default();
            writeln!(f, "  pattern {}: {}{direction}", p.number, p.label)?;
        }
        Ok(())
    }
}
