//! The filter "building kit": a project is an ordered list of named filters,
//! each of which consists of parts that can be switched off temporarily.
//!
//! ## Masks and how filters combine
//!
//! Every filter describes its *target* – the part of the sound it is about –
//! as a soft mask m(t,k) with values in 0..=1:
//!
//! * a noise-reduction filter targets the stationary background noise,
//! * an event filter targets the marked sound plus everything the analysis
//!   found to belong to it (each kind of evidence is a separately
//!   switchable part).
//!
//! The *action* decides what happens to the target:
//!
//! * **Suppress** multiplies the signal by the power complement
//!   `√(1 − m²)`, i.e. removes the target's share of the energy.
//! * **Extract** keeps only the target. All enabled extract filters are
//!   combined as a union (maximum), so extracting a bird and a frog keeps
//!   both.
//!
//! Final gain: `G = (max over extract masks, or 1 if there are none)
//!                · Π over suppress filters √(1 − m²)`.
//! Because suppress and extract of the same mask are power complements,
//! "suppress X" + "extract X" add up (in power) to the original.

use crate::analysis::{analyze_selection, AnalysisParams, RegionAnalysis, Selection};
use crate::noise::{NoiseEstimate, NoiseReduction};
use crate::stft::{Stft, StftConfig};
use crate::AudioData;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// What a filter does with its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FilterAction {
    /// Remove the target from the signal.
    #[default]
    Suppress,
    /// Keep only the target (union with other extract filters).
    Extract,
}

/// A filter built from a marked region of the spectrogram.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventFilter {
    /// The rectangle the user marked.
    pub selection: Selection,
    /// Tuning of the correlation search.
    #[serde(default)]
    pub params: AnalysisParams,
    /// Part switch: the energy inside the marked rectangle.
    pub use_selection: bool,
    /// Part switch: harmonically related partials.
    pub use_harmonics: bool,
    /// Part switch: co-modulated resonances.
    pub use_comodulation: bool,
    /// Analysis result; recomputed after loading (see [`Project::prepare`]).
    #[serde(skip)]
    pub analysis: Option<Arc<RegionAnalysis>>,
}

impl EventFilter {
    /// New event filter with all parts enabled and no analysis yet.
    pub fn new(selection: Selection) -> Self {
        Self {
            selection: selection.normalized(),
            params: AnalysisParams::default(),
            use_selection: true,
            use_harmonics: true,
            use_comodulation: true,
            analysis: None,
        }
    }
}

/// The kinds of filters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FilterKind {
    /// Stationary background noise, removed with a decision-directed soft
    /// gain.
    NoiseReduction(NoiseReduction),
    /// A marked sound and the parts of the spectrum that belong to it.
    Event(EventFilter),
}

/// One entry of the filter list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filter {
    /// User-visible name.
    pub name: String,
    /// Whether the filter affects the output.
    pub enabled: bool,
    /// Whether the filter's masks are drawn on top of the spectrogram.
    pub visible: bool,
    /// Suppress or extract.
    pub action: FilterAction,
    /// What the filter targets.
    pub kind: FilterKind,
}

impl Filter {
    /// A noise-reduction filter with default settings.
    pub fn noise_reduction(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            visible: false,
            action: FilterAction::Suppress,
            kind: FilterKind::NoiseReduction(NoiseReduction::default()),
        }
    }

    /// An event filter for `selection`.
    pub fn event(name: impl Into<String>, selection: Selection, action: FilterAction) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            visible: true,
            action,
            kind: FilterKind::Event(EventFilter::new(selection)),
        }
    }

    /// Names and on/off state of the filter's parts, in display order.
    pub fn parts(&self) -> Vec<(&'static str, bool)> {
        match &self.kind {
            FilterKind::NoiseReduction(_) => vec![("Background noise", true)],
            FilterKind::Event(e) => vec![
                ("Marked area", e.use_selection),
                ("Harmonics (follow the pitch)", e.use_harmonics),
                ("Co-modulated resonances", e.use_comodulation),
            ],
        }
    }

    /// Switches part `index` (as listed by [`Filter::parts`]) on or off.
    /// The single part of a noise-reduction filter cannot be switched; use
    /// [`Filter::enabled`] instead. Returns whether anything changed.
    pub fn set_part_enabled(&mut self, index: usize, on: bool) -> bool {
        let FilterKind::Event(e) = &mut self.kind else {
            return false;
        };
        let slot = match index {
            0 => &mut e.use_selection,
            1 => &mut e.use_harmonics,
            2 => &mut e.use_comodulation,
            _ => return false,
        };
        let changed = *slot != on;
        *slot = on;
        changed
    }
}

/// Target mask of an event filter at one frame: the maximum of its enabled
/// parts. Returns `None` outside the analysed time span or before analysis.
pub fn event_target(e: &EventFilter, frame: i64, out: &mut [f32]) -> bool {
    let Some(a) = &e.analysis else { return false };
    let Some((sel, harm, comod)) = a.frame(frame) else {
        return false;
    };
    for (k, o) in out.iter_mut().enumerate() {
        let mut m = 0.0f32;
        if e.use_selection {
            m = m.max(sel[k]);
        }
        if e.use_harmonics {
            m = m.max(harm[k]);
        }
        if e.use_comodulation {
            m = m.max(comod[k]);
        }
        *o = m;
    }
    true
}

/// Power complement √(1 − m²) of a mask value.
pub fn complement(m: f32) -> f32 {
    (1.0 - m * m).max(0.0).sqrt()
}

/// A complete processing setup, saved as JSON.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    /// File format version, for future migrations.
    pub version: u32,
    /// STFT used for all processing.
    pub stft: StftConfig,
    /// The filter list, in display order.
    pub filters: Vec<Filter>,
    /// The selection currently being edited in the GUI: drawn, never
    /// applied, never saved.
    #[serde(skip)]
    pub preview: Option<Filter>,
}

impl Default for Project {
    /// A project with one enabled background noise reduction filter.
    fn default() -> Self {
        Self {
            version: 1,
            stft: StftConfig::default(),
            filters: vec![Filter::noise_reduction("Background noise")],
            preview: None,
        }
    }
}

impl Project {
    /// Reads a project file.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
        let p: Project =
            serde_json::from_str(&text).with_context(|| format!("invalid project file {}", path.display()))?;
        p.stft.validate()?;
        Ok(p)
    }

    /// Writes the project as pretty-printed JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("cannot write {}", path.display()))
    }

    /// Noise estimate the analysis should use: that of the first enabled
    /// noise-reduction filter, so a learned profile also helps the search.
    pub fn analysis_noise(&self) -> NoiseEstimate {
        self.filters
            .iter()
            .find_map(|f| match (&f.kind, f.enabled) {
                (FilterKind::NoiseReduction(nr), true) => Some(nr.estimate.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Runs the selection analysis for every event filter (and the preview)
    /// that does not have one yet. Must be called before processing.
    pub fn prepare(&mut self, audio: &AudioData) {
        let mono = audio.mono();
        let stft = Stft::new(self.stft);
        let noise = self.analysis_noise();
        for f in self.filters.iter_mut().chain(self.preview.as_mut()) {
            if let FilterKind::Event(e) = &mut f.kind {
                if e.analysis.is_none() {
                    e.analysis = Some(Arc::new(analyze_selection(
                        &mono,
                        audio.sample_rate,
                        &stft,
                        &e.selection,
                        &noise,
                        &e.params,
                    )));
                }
            }
        }
    }
}

/// Per-frame masks produced by [`compute_gains`].
pub struct Gains {
    /// Gain to apply to each bin of each frame.
    pub gain: Vec<Vec<f32>>,
    /// Union of the "marked area" masks of visible filters (green).
    pub selected: Vec<Vec<f32>>,
    /// Union of the inferred masks of visible filters (red).
    pub inferred: Vec<Vec<f32>>,
}

/// Combines all filters of `project` into a gain per bin for frames
/// `first_frame .. first_frame + n`, given the mono power of frames
/// `first_frame - warmup .. first_frame + n` (the warm-up frames only prime
/// the recursive noise-reduction estimator) and the extra `context` power
/// frames used for adaptive noise estimation.
pub fn compute_gains(
    project: &Project,
    power_with_warmup: &[Vec<f32>],
    warmup: usize,
    context: &[Vec<f32>],
    first_frame: i64,
) -> Gains {
    let bins = project.stft.bins();
    let n = power_with_warmup.len() - warmup;
    let mut extract: Option<Vec<Vec<f32>>> = None;
    let mut suppress = vec![vec![1.0f32; bins]; n];
    let mut selected = vec![vec![0.0f32; bins]; n];
    let mut inferred = vec![vec![0.0f32; bins]; n];
    let mut scratch = vec![0.0f32; bins];

    for f in &project.filters {
        if !f.enabled {
            continue;
        }
        match &f.kind {
            FilterKind::NoiseReduction(nr) => {
                let noise = nr.noise_power(context, bins);
                let g = nr.gains(power_with_warmup, &noise);
                for (t, row) in g[warmup..].iter().enumerate() {
                    match f.action {
                        // complement(complement(g)) == g, so apply directly.
                        FilterAction::Suppress => row.iter().zip(&mut suppress[t]).for_each(|(g, s)| *s *= g),
                        FilterAction::Extract => {
                            let ex = extract.get_or_insert_with(|| vec![vec![0.0; bins]; n]);
                            row.iter().zip(&mut ex[t]).for_each(|(g, e)| *e = e.max(complement(*g)));
                        }
                    }
                }
            }
            FilterKind::Event(e) => {
                if f.action == FilterAction::Extract {
                    extract.get_or_insert_with(|| vec![vec![0.0; bins]; n]);
                }
                for t in 0..n {
                    if !event_target(e, first_frame + t as i64, &mut scratch) {
                        continue;
                    }
                    match f.action {
                        FilterAction::Suppress => scratch
                            .iter()
                            .zip(&mut suppress[t])
                            .for_each(|(m, s)| *s *= complement(*m)),
                        FilterAction::Extract => {
                            let ex = extract.as_mut().expect("created above");
                            scratch.iter().zip(&mut ex[t]).for_each(|(m, e)| *e = e.max(*m));
                        }
                    }
                }
            }
        }
    }

    // Overlays of visible event filters, including the live preview.
    for f in project.filters.iter().chain(project.preview.as_ref()) {
        let FilterKind::Event(e) = &f.kind else { continue };
        if !f.visible {
            continue;
        }
        let Some(a) = &e.analysis else { continue };
        for t in 0..n {
            let Some((sel, harm, comod)) = a.frame(first_frame + t as i64) else {
                continue;
            };
            for k in 0..bins {
                if e.use_selection {
                    selected[t][k] = selected[t][k].max(sel[k]);
                }
                let mut i = 0.0f32;
                if e.use_harmonics {
                    i = i.max(harm[k]);
                }
                if e.use_comodulation {
                    i = i.max(comod[k]);
                }
                inferred[t][k] = inferred[t][k].max(i);
            }
        }
    }

    let gain = match extract {
        Some(ex) => ex
            .into_iter()
            .zip(suppress)
            .map(|(e, s)| e.iter().zip(&s).map(|(a, b)| a * b).collect())
            .collect(),
        None => suppress,
    };
    Gains {
        gain,
        selected,
        inferred,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analysed_event(action: FilterAction, value: f32) -> Filter {
        let mut f = Filter::event(
            "e",
            Selection {
                start_s: 0.0,
                end_s: 1.0,
                low_hz: 0.0,
                high_hz: 1.0,
            },
            action,
        );
        if let FilterKind::Event(e) = &mut f.kind {
            e.analysis = Some(Arc::new(RegionAnalysis {
                first_frame: 0,
                frames: 2,
                bins: 3,
                selected: vec![vec![value, 0.0, 0.0]; 2],
                harmonics: vec![vec![0.0, value, 0.0]; 2],
                comodulated: vec![vec![0.0, 0.0, value]; 2],
                ..Default::default()
            }));
        }
        f
    }

    fn project(filters: Vec<Filter>) -> Project {
        Project {
            version: 1,
            stft: StftConfig { fft_size: 4, hop: 2 },
            filters,
            preview: None,
        }
    }

    #[test]
    fn suppress_and_extract_are_power_complements() {
        let power = vec![vec![1.0; 3]; 2];
        let s = compute_gains(
            &project(vec![analysed_event(FilterAction::Suppress, 0.6)]),
            &power,
            0,
            &power,
            0,
        );
        let e = compute_gains(
            &project(vec![analysed_event(FilterAction::Extract, 0.6)]),
            &power,
            0,
            &power,
            0,
        );
        for k in 0..3 {
            let sum = s.gain[0][k].powi(2) + e.gain[0][k].powi(2);
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn disabled_parts_and_filters_do_nothing() {
        let power = vec![vec![1.0; 3]; 2];
        let mut f = analysed_event(FilterAction::Suppress, 1.0);
        assert!(f.set_part_enabled(1, false));
        assert!(!f.set_part_enabled(1, false));
        let g = compute_gains(&project(vec![f.clone()]), &power, 0, &power, 0);
        assert_eq!(g.gain[0], vec![0.0, 1.0, 0.0]);
        f.enabled = false;
        let g = compute_gains(&project(vec![f]), &power, 0, &power, 0);
        assert_eq!(g.gain[0], vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn extract_filters_combine_as_union_and_outside_span_is_silent() {
        let power = vec![vec![1.0; 3]; 4];
        let mut a = analysed_event(FilterAction::Extract, 1.0);
        a.set_part_enabled(1, false);
        a.set_part_enabled(2, false);
        let mut b = analysed_event(FilterAction::Extract, 1.0);
        b.set_part_enabled(0, false);
        b.set_part_enabled(2, false);
        let g = compute_gains(&project(vec![a, b]), &power, 0, &power, 0);
        assert_eq!(g.gain[0], vec![1.0, 1.0, 0.0]);
        // Frames 2 and 3 lie outside the analysed span: nothing to extract.
        assert_eq!(g.gain[3], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn overlays_follow_visibility_not_enabled_state() {
        let power = vec![vec![1.0; 3]; 2];
        let mut f = analysed_event(FilterAction::Suppress, 0.8);
        f.enabled = false;
        let g = compute_gains(&project(vec![f.clone()]), &power, 0, &power, 0);
        assert_eq!(g.selected[0][0], 0.8);
        assert_eq!(g.inferred[0][1], 0.8);
        f.visible = false;
        let g = compute_gains(&project(vec![f]), &power, 0, &power, 0);
        assert_eq!(g.selected[0][0], 0.0);
    }

    #[test]
    fn project_json_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.json");
        let mut p = Project::default();
        p.filters.push(Filter::event(
            "bird",
            Selection {
                start_s: 1.0,
                end_s: 2.0,
                low_hz: 2000.0,
                high_hz: 4000.0,
            },
            FilterAction::Extract,
        ));
        p.filters[1].set_part_enabled(2, false);
        p.save(&path).unwrap();
        let q = Project::load(&path).unwrap();
        assert_eq!(q.filters.len(), 2);
        assert_eq!(q.filters[1].name, "bird");
        assert_eq!(q.filters[1].action, FilterAction::Extract);
        assert!(!q.filters[1].parts()[2].1);
    }
}
