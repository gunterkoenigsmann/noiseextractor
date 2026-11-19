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
//!   switchable part),
//! * a discovered filter targets a sound found automatically: its share of
//!   the spectral patterns learned by [`crate::discover`] (each pattern is
//!   a switchable part).
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
use crate::discover::{Decomposition, Discovery, FrameFit};
use crate::noise::{NoiseContext, NoiseEstimate, NoiseReduction};
use crate::similar::{add_occurrence_mask, Occurrence, SimilarParams};
use crate::spatial::Signature;
use crate::stft::{Stft, StftConfig};
use crate::AudioData;
use anyhow::{bail, Context, Result};
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
    /// Part switch: other occurrences of the same sound in the recording.
    /// Defaults to off for projects saved before this part existed.
    #[serde(default)]
    pub use_similar: bool,
    /// Tuning of the search for other occurrences.
    #[serde(default)]
    pub similar: SimilarParams,
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
            use_similar: true,
            similar: SimilarParams::default(),
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
    /// A sound found automatically: some of the spectral patterns of a
    /// [`Decomposition`] (see [`crate::discover`]).
    Discovered(DiscoveredFilter),
}

/// One spectral pattern of a [`DiscoveredFilter`]: a switchable part.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredPart {
    /// Index of the pattern in its decomposition.
    pub component: usize,
    /// What the pattern sounds like, e.g. "tonal, 1.9–3.1 kHz, peak 2.4 kHz".
    pub label: String,
    /// Part switch.
    pub enabled: bool,
}

/// A filter made of automatically found spectral patterns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredFilter {
    /// Index into [`Project::decompositions`].
    pub decomposition: usize,
    /// The patterns that make up the sound.
    pub parts: Vec<DiscoveredPart>,
}

impl DiscoveredFilter {
    /// Components of the enabled parts.
    pub fn enabled_components(&self) -> Vec<usize> {
        self.parts.iter().filter(|p| p.enabled).map(|p| p.component).collect()
    }
}

/// One entry of the filter list.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filter {
    /// User-visible name.
    pub name: String,
    /// Whether the filter affects the output.
    pub enabled: bool,
    /// Whether the filter's masks are drawn on top of the spectrogram: an
    /// event filter's marked area (green) and what was found to belong to
    /// it (red); for a noise-reduction filter, what it takes away (red).
    pub visible: bool,
    /// Suppress or extract.
    pub action: FilterAction,
    /// What the filter targets.
    pub kind: FilterKind,
    /// Suggested by the automatic search and not kept yet: the next search
    /// replaces it (see [`Project::add_discovery`]).
    #[serde(default)]
    pub suggested: bool,
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
            suggested: false,
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
            suggested: false,
        }
    }

    /// Names and on/off state of the filter's parts, in display order.
    pub fn parts(&self) -> Vec<(String, bool)> {
        let fixed = |parts: &[(&str, bool)]| parts.iter().map(|&(l, on)| (l.to_string(), on)).collect();
        match &self.kind {
            FilterKind::NoiseReduction(_) => fixed(&[("Background noise", true)]),
            FilterKind::Event(e) => fixed(&[
                ("Marked area", e.use_selection),
                ("Harmonics (follow the pitch)", e.use_harmonics),
                ("Co-modulated resonances", e.use_comodulation),
                ("Same sound elsewhere", e.use_similar),
            ]),
            FilterKind::Discovered(d) => d
                .parts
                .iter()
                .map(|p| (format!("Pattern {}: {}", p.component + 1, p.label), p.enabled))
                .collect(),
        }
    }

    /// Switches part `index` (as listed by [`Filter::parts`]) on or off.
    /// The single part of a noise-reduction filter cannot be switched; use
    /// [`Filter::enabled`] instead. Returns whether anything changed.
    pub fn set_part_enabled(&mut self, index: usize, on: bool) -> bool {
        let e = match &mut self.kind {
            FilterKind::Event(e) => e,
            FilterKind::Discovered(d) => {
                let Some(part) = d.parts.get_mut(index) else {
                    return false;
                };
                let changed = part.enabled != on;
                part.enabled = on;
                return changed;
            }
            FilterKind::NoiseReduction(_) => return false,
        };
        let slot = match index {
            0 => &mut e.use_selection,
            1 => &mut e.use_harmonics,
            2 => &mut e.use_comodulation,
            3 => &mut e.use_similar,
            _ => return false,
        };
        let changed = *slot != on;
        *slot = on;
        changed
    }
}

/// Masks of an event filter at global frame `frame`: the marked area goes
/// to `selected`, everything inferred (harmonics, co-modulation, other
/// occurrences) to `inferred`; only enabled parts contribute. `power` is the
/// mono power of that frame (needed to weight other occurrences) and
/// `occurrences` are the places where the sound was found again. Returns
/// `false` (leaving the outputs untouched) when nothing applies to the frame.
pub fn event_masks(
    e: &EventFilter,
    frame: i64,
    power: &[f32],
    occurrences: &[Occurrence],
    selected: &mut [f32],
    inferred: &mut [f32],
) -> bool {
    let Some(a) = &e.analysis else { return false };
    let mut any = false;
    if let Some((sel, harm, comod)) = a.frame(frame) {
        any = true;
        for k in 0..selected.len() {
            selected[k] = if e.use_selection { sel[k] } else { 0.0 };
            let mut m = 0.0f32;
            if e.use_harmonics {
                m = m.max(harm[k]);
            }
            if e.use_comodulation {
                m = m.max(comod[k]);
            }
            inferred[k] = m;
        }
    }
    if e.use_similar {
        if let Some(tpl) = &a.template {
            // Membership of a template cell: the enabled parts of the
            // original, so switching off e.g. harmonics also removes them
            // from every copy.
            let member = |rel: usize, k: usize| -> f32 {
                let Some((sel, harm, comod)) = a.frame(tpl.first_frame + rel as i64) else {
                    return 0.0;
                };
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
                m
            };
            // The original is already covered by the parts above; matching
            // it again would only turn the green marked area orange.
            let relevant = occurrences.iter().filter(|o| o.applies_to(tpl, frame));
            for occ in relevant {
                if !any {
                    selected.iter_mut().for_each(|v| *v = 0.0);
                    inferred.iter_mut().for_each(|v| *v = 0.0);
                    any = true;
                }
                add_occurrence_mask(tpl, occ, &e.similar, frame, power, &member, inferred);
            }
        }
    }
    any
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
    /// Index of the filter picked in the GUI's list: drawn like a visible
    /// filter, so picking a filter shows what it does. Never saved.
    #[serde(skip)]
    pub highlight: Option<usize>,
    /// Spectral patterns found by automatic searches, referred to by
    /// [`DiscoveredFilter::decomposition`].
    #[serde(default)]
    pub decompositions: Vec<Decomposition>,
}

impl Default for Project {
    /// A project with one enabled background noise reduction filter.
    fn default() -> Self {
        Self {
            version: 1,
            stft: StftConfig::default(),
            filters: vec![Filter::noise_reduction("Background noise")],
            preview: None,
            highlight: None,
            decompositions: Vec::new(),
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
        // Patterns no filter uses any more are not worth saving.
        let mut tidy = self.clone();
        tidy.prune_decompositions();
        std::fs::write(path, serde_json::to_string_pretty(&tidy)?)
            .with_context(|| format!("cannot write {}", path.display()))
    }

    /// Whether the overlays of filter `index` are drawn: when it is
    /// visible or highlighted. Index `filters.len()` is the preview, which
    /// is always drawn.
    pub fn shows(&self, index: usize) -> bool {
        match self.filters.get(index) {
            Some(f) => f.visible || self.highlight == Some(index),
            None => index == self.filters.len() && self.preview.is_some(),
        }
    }

    /// The filter whose found sound [`Gains::found`] measures: the
    /// selection being edited (index `filters.len()`) if there is one,
    /// else the highlighted filter. That is the sound the user is working
    /// on, so the GUI's overview can show where in the recording, and how
    /// loud, the program found it.
    pub fn found_target(&self) -> Option<usize> {
        if self.preview.is_some() {
            Some(self.filters.len())
        } else {
            self.highlight.filter(|&i| i < self.filters.len())
        }
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
                        mono,
                        &audio.channels,
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

    /// Adds the result of an automatic search: its patterns, and one
    /// suggested filter per group, switched off and hidden (picking one in
    /// the list shows it). Suggestions of earlier searches that were not
    /// kept are replaced. Returns the index of the first new filter.
    pub fn add_discovery(&mut self, discovery: Discovery) -> usize {
        self.discard_suggestions();
        let index = self.decompositions.len();
        let first = self.filters.len();
        for (n, group) in discovery.groups.iter().enumerate() {
            let label = |c: usize| discovery.decomposition.labels.get(c).cloned().unwrap_or_default();
            self.filters.push(Filter {
                name: format!("Found {}: {}", n + 1, label(group[0])),
                enabled: false,
                visible: false,
                action: FilterAction::Suppress,
                kind: FilterKind::Discovered(DiscoveredFilter {
                    decomposition: index,
                    parts: group
                        .iter()
                        .map(|&c| DiscoveredPart {
                            component: c,
                            label: label(c),
                            enabled: true,
                        })
                        .collect(),
                }),
                suggested: true,
            });
        }
        self.decompositions.push(discovery.decomposition);
        first
    }

    /// Removes all suggestions that were not kept; returns how many.
    pub fn discard_suggestions(&mut self) -> usize {
        let before = self.filters.len();
        self.filters.retain(|f| !f.suggested);
        self.prune_decompositions();
        before - self.filters.len()
    }

    /// Merges filter `from` into filter `into`: both must be found by the
    /// same search. `into` gets the parts of both (and counts as kept);
    /// `from` is removed. Returns the new index of the merged filter.
    pub fn merge(&mut self, into: usize, from: usize) -> Result<usize> {
        let decomposition_of = |f: Option<&Filter>| match f.map(|f| &f.kind) {
            Some(FilterKind::Discovered(d)) => Some(d.decomposition),
            _ => None,
        };
        let (Some(a), Some(b)) = (
            decomposition_of(self.filters.get(into)),
            decomposition_of(self.filters.get(from)),
        ) else {
            bail!("only automatically found sounds can be merged");
        };
        if into == from {
            bail!("pick two different sounds to merge");
        }
        if a != b {
            bail!("only sounds found by the same search can be merged");
        }
        let removed = self.filters.remove(from);
        let into = if from < into { into - 1 } else { into };
        if let (FilterKind::Discovered(dst), FilterKind::Discovered(src)) = (&mut self.filters[into].kind, removed.kind)
        {
            for part in src.parts {
                if !dst.parts.iter().any(|p| p.component == part.component) {
                    dst.parts.push(part);
                }
            }
            dst.parts.sort_by_key(|p| p.component);
        }
        self.filters[into].suggested = false;
        Ok(into)
    }

    /// The filters that can be merged into filter `index`: other sounds
    /// found by the same search.
    pub fn merge_candidates(&self, index: usize) -> Vec<usize> {
        let Some(FilterKind::Discovered(d)) = self.filters.get(index).map(|f| &f.kind) else {
            return Vec::new();
        };
        self.filters
            .iter()
            .enumerate()
            .filter(|&(j, f)| {
                j != index && matches!(&f.kind, FilterKind::Discovered(o) if o.decomposition == d.decomposition)
            })
            .map(|(j, _)| j)
            .collect()
    }

    /// Splits an automatically found sound into one filter per pattern,
    /// in its place. Returns how many filters it became.
    pub fn split(&mut self, index: usize) -> Result<usize> {
        let Some(Filter {
            kind: FilterKind::Discovered(d),
            ..
        }) = self.filters.get(index)
        else {
            bail!("only automatically found sounds can be split");
        };
        if d.parts.len() < 2 {
            bail!("this sound has only one pattern");
        }
        let original = self.filters.remove(index);
        let FilterKind::Discovered(d) = &original.kind else {
            unreachable!("checked above")
        };
        let pieces: Vec<Filter> = d
            .parts
            .iter()
            .map(|part| Filter {
                name: format!("{} / pattern {}", original.name, part.component + 1),
                kind: FilterKind::Discovered(DiscoveredFilter {
                    decomposition: d.decomposition,
                    parts: vec![DiscoveredPart {
                        enabled: true,
                        ..part.clone()
                    }],
                }),
                ..original.clone()
            })
            .collect();
        let n = pieces.len();
        self.filters.splice(index..index, pieces);
        Ok(n)
    }

    /// Drops decompositions no filter refers to and renumbers the rest.
    pub fn prune_decompositions(&mut self) {
        let used: Vec<bool> = (0..self.decompositions.len())
            .map(|i| {
                self.filters
                    .iter()
                    .any(|f| matches!(&f.kind, FilterKind::Discovered(d) if d.decomposition == i))
            })
            .collect();
        let mut new_index = vec![0; used.len()];
        let mut next = 0;
        for (i, &u) in used.iter().enumerate() {
            new_index[i] = next;
            next += usize::from(u);
        }
        let mut i = 0;
        self.decompositions.retain(|_| {
            i += 1;
            used[i - 1]
        });
        for f in &mut self.filters {
            if let FilterKind::Discovered(d) = &mut f.kind {
                if let Some(&n) = new_index.get(d.decomposition) {
                    d.decomposition = n;
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
    /// Power (Σ over bins of `P · m²`) of what the analysis found of the
    /// [`Project::found_target`] alone, per frame: the energy under that
    /// filter's red and orange overlay. Empty when there is no target.
    pub found: Vec<f32>,
}

/// Combines all filters of `project` into a gain per bin for frames
/// `first_frame .. first_frame + n`, given the mono power of frames
/// `first_frame - warmup .. first_frame + n` (the warm-up frames only prime
/// the recursive noise-reduction estimator) and the `context` power frames
/// the adaptive noise estimate is taken from (it should cover
/// [`NoiseContext::needed`] for the frames `first_frame - warmup ..
/// first_frame + n`).
///
/// `occurrences[i]` lists where the sound of filter `i` was found again
/// (index `filters.len()` is the preview); missing entries mean none.
///
/// Where an extract and a suppress filter claim the same bin, the bin's
/// energy is shared out between them (see [`crate::share`]); `channels`
/// are the spectra of the individual channels for the same frames
/// (`first_frame ..`), used there to tell sounds apart by direction. Pass
/// `&[]` for mono recordings or when they are not at hand.
#[allow(clippy::too_many_arguments)]
pub fn compute_gains(
    project: &Project,
    power_with_warmup: &[Vec<f32>],
    warmup: usize,
    context: &NoiseContext,
    first_frame: i64,
    occurrences: &[Vec<Occurrence>],
    channels: &[crate::stft::Spectra],
) -> Gains {
    let bins = project.stft.bins();
    let n = power_with_warmup.len() - warmup;
    let mut extract: Option<Vec<Vec<f32>>> = None;
    let mut suppress = vec![vec![1.0f32; bins]; n];
    let mut selected = vec![vec![0.0f32; bins]; n];
    let mut inferred = vec![vec![0.0f32; bins]; n];
    let mut scratch_sel = vec![0.0f32; bins];
    let mut scratch_inf = vec![0.0f32; bins];
    let no_occurrences: Vec<Occurrence> = Vec::new();
    let occ_of = |i: usize| occurrences.get(i).unwrap_or(&no_occurrences);
    let fits = fit_decompositions(project, &power_with_warmup[warmup..]);
    let target = project.found_target();
    let mut found = if target.is_some() { vec![0.0f32; n] } else { Vec::new() };
    // Adds the energy of `power` under the mask `m` to `found[t]` (`m` is
    // an amplitude gain, so the energy it selects is P·m²).
    let add_found = |found: &mut [f32], t: usize, power: &[f32], m: &[f32]| {
        found[t] += power.iter().zip(m).map(|(p, m)| p * m * m).sum::<f32>();
    };
    // The same for what a noise-reduction filter with gains `g` (rows
    // with warm-up) removes: its power complement, 1 − g².
    let add_found_removed = |found: &mut [f32], g: &[Vec<f32>]| {
        for (t, (row, power)) in g[warmup..].iter().zip(&power_with_warmup[warmup..]).enumerate() {
            found[t] += power
                .iter()
                .zip(row)
                .map(|(p, g)| p * (1.0 - g * g).max(0.0))
                .sum::<f32>();
        }
    };

    // Event filters' masks first (frame by frame), so that bins claimed by
    // both an extract and a suppress filter can be shared out before the
    // masks are applied.
    let mut events: EventMasks = Vec::new();
    // Noise per frame (without the warm-up) of the first enabled noise
    // reduction, for sharing bins.
    let mut share_noise: Option<Vec<Vec<f32>>> = None;
    // Noise per frame of `power_with_warmup` for a noise-reduction filter.
    let track = |nr: &crate::noise::NoiseReduction| {
        nr.noise_track(context, first_frame - warmup as i64, power_with_warmup.len(), bins)
    };
    for (fi, f) in project.filters.iter().enumerate() {
        if !f.enabled {
            // A disabled noise reduction still shows what it would take away.
            if let FilterKind::NoiseReduction(nr) = &f.kind {
                if project.shows(fi) {
                    let g = nr.gains_tracked(power_with_warmup, &track(nr));
                    add_removed(&g[warmup..], &mut inferred);
                    if target == Some(fi) {
                        add_found_removed(&mut found, &g);
                    }
                }
            }
            continue;
        }
        match &f.kind {
            FilterKind::NoiseReduction(nr) => {
                let noise = track(nr);
                let g = nr.gains_tracked(power_with_warmup, &noise);
                if project.shows(fi) {
                    add_removed(&g[warmup..], &mut inferred);
                }
                if target == Some(fi) {
                    add_found_removed(&mut found, &g);
                }
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
                share_noise.get_or_insert_with(|| noise[warmup..].to_vec());
            }
            FilterKind::Event(e) => {
                if f.action == FilterAction::Extract {
                    extract.get_or_insert_with(|| vec![vec![0.0; bins]; n]);
                }
                let masks = (0..n)
                    .map(|t| {
                        let frame = first_frame + t as i64;
                        let power = &power_with_warmup[warmup + t];
                        event_masks(e, frame, power, occ_of(fi), &mut scratch_sel, &mut scratch_inf)
                            .then(|| (0..bins).map(|k| scratch_sel[k].max(scratch_inf[k])).collect())
                    })
                    .collect();
                events.push((fi, f, Source::Event(e), masks));
            }
            FilterKind::Discovered(d) => {
                let Some((dec, fits)) = discovered(project, &fits, d) else {
                    continue;
                };
                if f.action == FilterAction::Extract {
                    extract.get_or_insert_with(|| vec![vec![0.0; bins]; n]);
                }
                let parts = d.enabled_components();
                if parts.is_empty() {
                    continue;
                }
                let mut masks = Vec::with_capacity(n);
                let mut shapes = Vec::with_capacity(n);
                for fit in fits {
                    let mut shape = vec![0.0f32; bins];
                    dec.expected_power(fit, &parts, &mut shape);
                    let mut mask = vec![0.0f32; bins];
                    dec.mask(fit, &parts, &mut mask);
                    masks.push(Some(mask));
                    shapes.push(shape);
                }
                // The strongest enabled pattern's direction stands for the
                // sound's.
                let direction = parts
                    .iter()
                    .find_map(|&c| dec.directions.get(c)?.as_ref())
                    .map(|sig| (sig, dec.level_tolerance_db, dec.delay_tolerance_samples));
                events.push((fi, f, Source::Discovered { shapes, direction }, masks));
            }
        }
    }
    let both = |extract: bool| {
        events
            .iter()
            .any(|(_, f, _, _)| (f.action == FilterAction::Extract) == extract)
    };
    if both(true) && both(false) {
        let noise = share_noise.unwrap_or_else(|| {
            events
                .iter()
                .find_map(|(_, _, src, _)| match src {
                    Source::Event(e) => e.analysis.as_ref()?.template.as_ref().map(|t| vec![t.noise.clone(); n]),
                    Source::Discovered { .. } => None,
                })
                .unwrap_or_else(|| vec![vec![0.0; bins]; n])
        });
        share_bins(
            &mut events,
            &power_with_warmup[warmup..],
            &noise,
            first_frame,
            occurrences,
            channels,
            project.stft.fft_size,
        );
    }
    for (_, f, _, masks) in &events {
        for (t, m) in masks.iter().enumerate() {
            let Some(m) = m else { continue };
            match f.action {
                FilterAction::Suppress => m.iter().zip(&mut suppress[t]).for_each(|(m, s)| *s *= complement(*m)),
                FilterAction::Extract => {
                    let ex = extract.as_mut().expect("created above");
                    m.iter().zip(&mut ex[t]).for_each(|(m, e)| *e = e.max(*m));
                }
            }
        }
    }

    // Overlays of visible (or highlighted) event filters, including the
    // live preview. Automatically found sounds are what the program
    // detected, so they are drawn in red.
    for (fi, f) in project.filters.iter().chain(project.preview.as_ref()).enumerate() {
        if !project.shows(fi) {
            continue;
        }
        let e = match &f.kind {
            FilterKind::Event(e) => e,
            FilterKind::Discovered(d) => {
                if let Some((dec, fits)) = discovered(project, &fits, d) {
                    let parts = d.enabled_components();
                    for (t, (fit, inf)) in fits.iter().zip(inferred.iter_mut()).enumerate() {
                        dec.mask(fit, &parts, &mut scratch_inf);
                        inf.iter_mut().zip(&scratch_inf).for_each(|(i, m)| *i = i.max(*m));
                        if target == Some(fi) {
                            add_found(&mut found, t, &power_with_warmup[warmup + t], &scratch_inf);
                        }
                    }
                }
                continue;
            }
            FilterKind::NoiseReduction(_) => continue,
        };
        for t in 0..n {
            let power = &power_with_warmup[warmup + t];
            if !event_masks(
                e,
                first_frame + t as i64,
                power,
                occ_of(fi),
                &mut scratch_sel,
                &mut scratch_inf,
            ) {
                continue;
            }
            for k in 0..bins {
                selected[t][k] = selected[t][k].max(scratch_sel[k]);
                inferred[t][k] = inferred[t][k].max(scratch_inf[k]);
            }
            if target == Some(fi) {
                add_found(&mut found, t, power, &scratch_inf);
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
        found,
    }
}

/// Adds what a noise-reduction filter with gains `g` (one row per frame)
/// takes away to the red overlay: its target, the noise, is the power
/// complement of the gain (see the module documentation), whether the
/// filter suppresses it or keeps only it.
fn add_removed(g: &[Vec<f32>], inferred: &mut [Vec<f32>]) {
    for (row, inf) in g.iter().zip(inferred.iter_mut()) {
        for (g, i) in row.iter().zip(inf.iter_mut()) {
            *i = i.max(complement(*g));
        }
    }
}

/// Fits every decomposition that an enabled or drawn filter uses to each
/// of the `power` frames (see [`Decomposition::fit_frame`]); `None` for
/// the others and for decompositions learned with another FFT size.
fn fit_decompositions(project: &Project, power: &[Vec<f32>]) -> Vec<Option<Vec<FrameFit>>> {
    let bins = project.stft.bins();
    (0..project.decompositions.len())
        .map(|i| {
            let dec = &project.decompositions[i];
            let needed = project.filters.iter().enumerate().any(|(fi, f)| {
                matches!(&f.kind, FilterKind::Discovered(d) if d.decomposition == i) && (f.enabled || project.shows(fi))
            });
            (needed && dec.bins() == bins && dec.fft_size == project.stft.fft_size)
                .then(|| power.iter().map(|p| dec.fit_frame(p)).collect())
        })
        .collect()
}

/// The decomposition of `d` with its frame fits, if they were computed.
fn discovered<'a>(
    project: &'a Project,
    fits: &'a [Option<Vec<FrameFit>>],
    d: &DiscoveredFilter,
) -> Option<(&'a Decomposition, &'a Vec<FrameFit>)> {
    Some((
        project.decompositions.get(d.decomposition)?,
        fits.get(d.decomposition)?.as_ref()?,
    ))
}

/// What the bin sharing knows about a filter's sound.
enum Source<'a> {
    /// A marked sound: its shape comes from the analysis' source model.
    Event(&'a EventFilter),
    /// An automatically found sound: the patterns give its expected power
    /// per frame directly, and the direction of its strongest pattern.
    Discovered {
        /// Expected power per bin in every frame.
        shapes: Vec<Vec<f32>>,
        /// Direction with level and delay tolerances, if known.
        direction: Option<(&'a Signature, f32, f32)>,
    },
}

/// Event and discovered filters with their masks per frame, as collected
/// in [`compute_gains`]: `(index in the list, filter, source, masks)`.
type EventMasks<'a> = Vec<(usize, &'a Filter, Source<'a>, Vec<Option<Vec<f32>>>)>;

/// Shares contested bins between extract and suppress filters (see
/// [`crate::share`]): predicts each sound's shape in every frame from its
/// source model, measures its loudness on its uncontested bins (borrowing
/// from neighbouring frames where none is free), and replaces the masks
/// in contested bins by the shares. In contested bins the suppress masks
/// are then dropped, because the extract masks already hold only what is
/// to be kept.
fn share_bins(
    events: &mut EventMasks,
    power: &[Vec<f32>],
    noise: &[Vec<f32>],
    first_frame: i64,
    occurrences: &[Vec<Occurrence>],
    channels: &[crate::stft::Spectra],
    fft_size: usize,
) {
    use crate::share::{contested, fill_gaps, loudness, share_frame, Claimant};
    let n = power.len();
    let bins = power.first().map_or(0, Vec::len);
    // Contested bins per frame (from the masks alone).
    let contested_bins: Vec<Option<Vec<bool>>> = (0..n)
        .map(|t| {
            let masks: Vec<(bool, &[f32])> = events
                .iter()
                .filter_map(|(_, f, _, m)| Some((f.action == FilterAction::Extract, m[t].as_deref()?)))
                .collect();
            let c = contested(&masks);
            c.iter().any(|&b| b).then_some(c)
        })
        .collect();
    if contested_bins.iter().all(Option::is_none) {
        return;
    }
    // Predicted shapes and loudness of every filter's sound per frame.
    let mut shapes: Vec<Vec<Option<Vec<f32>>>> = Vec::with_capacity(events.len());
    let mut loud: Vec<Vec<Option<f32>>> = Vec::with_capacity(events.len());
    let mut scratch = vec![0.0f32; bins];
    for (fi, _, src, masks) in events.iter() {
        let e = match src {
            Source::Event(e) => e,
            Source::Discovered { shapes: s, .. } => {
                // The patterns predict the power itself: loudness 1.
                shapes.push((0..n).map(|t| masks[t].as_ref().map(|_| s[t].clone())).collect());
                loud.push((0..n).map(|t| masks[t].as_ref().map(|_| 1.0)).collect());
                continue;
            }
        };
        let a = e.analysis.as_ref();
        let model = a.and_then(|a| a.model.as_ref());
        let (mut sh, mut lo) = (vec![None; n], vec![None; n]);
        if let (Some(a), Some(model)) = (a, model) {
            for t in 0..n {
                if masks[t].is_none() {
                    continue;
                }
                let frame = first_frame + t as i64;
                let mut placements = Vec::new();
                if a.frame(frame).is_some() {
                    placements.push(((frame - a.first_frame) as usize, 0.0f32));
                }
                if let (true, Some(tpl)) = (e.use_similar, &a.template) {
                    for o in occurrences.get(*fi).map_or(&[][..], Vec::as_slice) {
                        if o.applies_to(tpl, frame) {
                            let idx = tpl.first_frame - a.first_frame + o.template_frame(tpl, frame) as i64;
                            placements.push((idx as usize, o.shift as f32));
                        }
                    }
                }
                if placements.is_empty() {
                    continue;
                }
                let mut shape = vec![0.0f32; bins];
                for (idx, shift) in placements {
                    let pitch = a.pitch_track.get(idx).copied().flatten();
                    model.shape(pitch, shift, &mut scratch);
                    shape.iter_mut().zip(&scratch).for_each(|(s, v)| *s += v);
                }
                let free = vec![false; bins];
                let c = contested_bins[t].as_deref().unwrap_or(&free);
                lo[t] = loudness(&shape, &power[t], &noise[t], c);
                sh[t] = Some(shape);
            }
            fill_gaps(&mut lo);
        }
        shapes.push(sh);
        loud.push(lo);
    }
    // Share every contested bin.
    for t in 0..n {
        let Some(c) = &contested_bins[t] else { continue };
        let mut claimants: Vec<Claimant> = Vec::new();
        for (i, (_, f, src, masks)) in events.iter_mut().enumerate() {
            let Some(mask) = masks[t].as_mut() else { continue };
            let direction = match src {
                Source::Event(e) => e
                    .analysis
                    .as_ref()
                    .and_then(|a| a.template.as_ref())
                    .and_then(|tpl| tpl.spatial.as_ref())
                    .map(|sig| (sig, e.similar.level_tolerance_db, e.similar.delay_tolerance_samples)),
                Source::Discovered { direction, .. } => *direction,
            };
            claimants.push(Claimant {
                extract: f.action == FilterAction::Extract,
                mask: mask.as_mut_slice(),
                shape: shapes[i][t].clone(),
                loudness: loud[i][t],
                direction,
            });
        }
        share_frame(&mut claimants, c, &power[t], &noise[t], channels, t, fft_size);
        // The extract shares hold what is kept; suppressing again would
        // remove the kept share a second time.
        for cl in claimants.iter_mut().filter(|cl| !cl.extract) {
            for k in (0..bins).filter(|&k| c[k]) {
                cl.mask[k] = 0.0;
            }
        }
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
            highlight: None,
            decompositions: Vec::new(),
        }
    }

    #[test]
    fn suppress_and_extract_are_power_complements() {
        let power = vec![vec![1.0; 3]; 2];
        let s = compute_gains(
            &project(vec![analysed_event(FilterAction::Suppress, 0.6)]),
            &power,
            0,
            &NoiseContext::new(&power, 0),
            0,
            &[],
            &[],
        );
        let e = compute_gains(
            &project(vec![analysed_event(FilterAction::Extract, 0.6)]),
            &power,
            0,
            &NoiseContext::new(&power, 0),
            0,
            &[],
            &[],
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
        let g = compute_gains(
            &project(vec![f.clone()]),
            &power,
            0,
            &NoiseContext::new(&power, 0),
            0,
            &[],
            &[],
        );
        assert_eq!(g.gain[0], vec![0.0, 1.0, 0.0]);
        f.enabled = false;
        let g = compute_gains(&project(vec![f]), &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
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
        let g = compute_gains(
            &project(vec![a, b]),
            &power,
            0,
            &NoiseContext::new(&power, 0),
            0,
            &[],
            &[],
        );
        assert_eq!(g.gain[0], vec![1.0, 1.0, 0.0]);
        // Frames 2 and 3 lie outside the analysed span: nothing to extract.
        assert_eq!(g.gain[3], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn overlays_follow_visibility_not_enabled_state() {
        let power = vec![vec![1.0; 3]; 2];
        let mut f = analysed_event(FilterAction::Suppress, 0.8);
        f.enabled = false;
        let g = compute_gains(
            &project(vec![f.clone()]),
            &power,
            0,
            &NoiseContext::new(&power, 0),
            0,
            &[],
            &[],
        );
        assert_eq!(g.selected[0][0], 0.8);
        assert_eq!(g.inferred[0][1], 0.8);
        f.visible = false;
        let g = compute_gains(&project(vec![f]), &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
        assert_eq!(g.selected[0][0], 0.0);
    }

    #[test]
    fn a_highlighted_filter_is_drawn_although_hidden() {
        // Picking a filter in the GUI's list highlights it; that must show
        // its overlays even when "Show in spectrogram" is off.
        let power = vec![vec![1.0; 3]; 2];
        let mut hidden = analysed_event(FilterAction::Suppress, 0.8);
        hidden.visible = false;
        let mut p = project(vec![Filter::noise_reduction("noise"), hidden]);
        let g = compute_gains(&p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
        assert_eq!((g.selected[0][0], g.inferred[0][1]), (0.0, 0.0));
        p.highlight = Some(1);
        let g = compute_gains(&p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
        assert_eq!((g.selected[0][0], g.inferred[0][1]), (0.8, 0.8));
        assert!(p.shows(1) && !p.shows(0) && !p.shows(2), "no preview, nothing at 2");
    }

    #[test]
    fn noise_reduction_overlay_shows_what_it_removes() {
        // Stationary noise in every bin: the filter takes most of it away,
        // and the red overlay must say so (it used to stay empty for
        // noise-reduction filters, visible or not).
        let power = vec![vec![1.0; 3]; 40];
        let mut p = project(vec![Filter::noise_reduction("noise")]);
        let removed = |p: &Project| {
            let g = compute_gains(p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
            let expected = complement(g.gain[39][0]);
            (g.inferred[39][0], expected)
        };
        assert_eq!(removed(&p).0, 0.0, "hidden and not highlighted");
        p.highlight = Some(0);
        let (shown, expected) = removed(&p);
        assert!(expected > 0.5, "test premise: noise is mostly removed ({expected})");
        assert_eq!(shown, expected);
        // Switched off, it still shows what it would remove.
        p.filters[0].enabled = false;
        p.highlight = None;
        p.filters[0].visible = true;
        assert!(removed(&p).0 > 0.5);
    }

    #[test]
    fn found_measures_the_picked_filter_alone() {
        // Bins with power 1, 2 and 4: filter 1 infers bins 1 and 2 with
        // 0.5, filter 2 with 0.8, so only the target's findings count.
        let power = vec![vec![1.0, 2.0, 4.0]; 2];
        let mut p = project(vec![
            Filter::noise_reduction("noise"),
            analysed_event(FilterAction::Suppress, 0.5),
            analysed_event(FilterAction::Suppress, 0.8),
        ]);
        let found = |p: &Project| compute_gains(p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]).found;
        assert!(found(&p).is_empty(), "nothing picked");
        p.highlight = Some(1);
        assert_eq!(found(&p), vec![0.25 * (2.0 + 4.0); 2]);
        p.highlight = Some(2);
        let f = found(&p);
        assert!((f[0] - 0.64 * 6.0).abs() < 1e-5, "{f:?}");
        // The selection being edited wins over the picked filter.
        p.preview = Some(analysed_event(FilterAction::Suppress, 1.0));
        assert_eq!(p.found_target(), Some(3));
        assert_eq!(found(&p), vec![6.0; 2]);
    }

    #[test]
    fn found_of_a_noise_reduction_is_what_it_removes() {
        let power = vec![vec![1.0, 2.0, 4.0]; 40];
        let mut p = project(vec![Filter::noise_reduction("noise")]);
        p.highlight = Some(0);
        for enabled in [true, false] {
            p.filters[0].enabled = enabled;
            let g = compute_gains(&p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
            // Removed power: P·(1 − g²) per bin.
            let removed: f32 = (0..3).map(|k| power[39][k] * (1.0 - g.gain[39][k].powi(2))).sum();
            let removed = if enabled {
                removed
            } else {
                // Switched off it removes nothing, but still shows what it
                // would: its overlay.
                (0..3).map(|k| power[39][k] * g.inferred[39][k].powi(2)).sum()
            };
            assert!(removed > 3.0, "test premise: most noise removed ({removed})");
            assert!((g.found[39] - removed).abs() < 1e-4, "{} vs {removed}", g.found[39]);
        }
    }

    /// A search result over 3 bins: pattern 0 lives in bin 0, pattern 1
    /// in bin 2, pattern 2 in bin 1; grouped as {0, 2} and {1}.
    fn discovery() -> Discovery {
        Discovery {
            decomposition: Decomposition {
                fft_size: 4,
                patterns: vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]],
                background: Vec::new(),
                directions: vec![None; 3],
                labels: vec!["low".into(), "high".into(), "middle".into()],
                level_tolerance_db: 3.0,
                delay_tolerance_samples: 2.0,
            },
            groups: vec![vec![0, 2], vec![1]],
            energy: vec![0.5, 0.3, 0.2],
            frames_used: 10,
        }
    }

    #[test]
    fn discovered_sounds_are_suggested_switched_off_and_hidden() {
        let mut p = project(vec![Filter::noise_reduction("noise")]);
        assert_eq!(p.add_discovery(discovery()), 1);
        assert_eq!(p.filters.len(), 3);
        let f = &p.filters[1];
        assert!(f.suggested && !f.enabled && !f.visible);
        assert_eq!(f.name, "Found 1: low");
        assert_eq!(f.parts().len(), 2);
        assert_eq!(f.parts()[1].0, "Pattern 3: middle");
        // Nothing changes the sound until a suggestion is switched on.
        let power = vec![vec![4.0, 4.0, 4.0]; 2];
        p.filters.remove(0);
        let g = compute_gains(&p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
        assert_eq!(g.gain[0], vec![1.0; 3]);
        assert_eq!(g.inferred[0], vec![0.0; 3]);
        // Picking it shows what it found, in red.
        p.highlight = Some(0);
        let g = compute_gains(&p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
        assert!(g.inferred[0][0] > 0.99 && g.inferred[0][1] > 0.99 && g.inferred[0][2] < 1e-3);
        assert_eq!(g.selected[0], vec![0.0; 3]);
    }

    #[test]
    fn discovered_sounds_suppress_and_extract_by_their_share() {
        let mut p = project(Vec::new());
        p.add_discovery(discovery());
        let power = vec![vec![9.0, 1.0, 4.0]; 2];
        p.filters[0].enabled = true;
        let s = compute_gains(&p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
        p.filters[0].action = FilterAction::Extract;
        let e = compute_gains(&p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
        for k in 0..3 {
            assert!((s.gain[0][k].powi(2) + e.gain[0][k].powi(2) - 1.0).abs() < 1e-5);
        }
        // Each bin belongs to exactly one pattern here.
        assert!(
            e.gain[0][0] > 0.99 && e.gain[0][1] > 0.99 && e.gain[0][2] < 1e-3,
            "{:?}",
            e.gain[0]
        );
        // Switching a part off gives its bins back.
        assert!(p.filters[0].set_part_enabled(1, false));
        let e = compute_gains(&p, &power, 0, &NoiseContext::new(&power, 0), 0, &[], &[]);
        assert!(e.gain[0][1] < 1e-3, "{:?}", e.gain[0]);
    }

    #[test]
    fn suggestions_are_merged_split_kept_and_replaced() {
        let mut p = project(Vec::new());
        p.add_discovery(discovery());
        assert!(p.merge(0, 0).is_err());
        assert_eq!(p.merge_candidates(0), vec![1]);
        assert_eq!(p.merge(1, 0).unwrap(), 0);
        assert_eq!(p.filters.len(), 1);
        let FilterKind::Discovered(d) = &p.filters[0].kind else {
            panic!()
        };
        assert_eq!(d.parts.iter().map(|p| p.component).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!(!p.filters[0].suggested, "merging is a decision to keep");
        assert_eq!(p.split(0).unwrap(), 3);
        assert_eq!(p.filters.len(), 3);
        assert!(p.split(0).is_err(), "one pattern cannot be split");
        // Another search replaces suggestions but keeps what was kept.
        p.add_discovery(discovery());
        assert_eq!(p.filters.len(), 5);
        assert_eq!(p.decompositions.len(), 2);
        // Sounds of different searches cannot be merged.
        assert!(p.merge(0, 3).is_err());
        assert_eq!(p.discard_suggestions(), 2);
        assert_eq!(p.decompositions.len(), 1);
        // Removing the last user of a decomposition renumbers the rest.
        p.filters.clear();
        p.add_discovery(discovery());
        p.filters.iter_mut().for_each(|f| f.suggested = false);
        p.add_discovery(discovery());
        p.filters.drain(0..2);
        p.prune_decompositions();
        assert_eq!(p.decompositions.len(), 1);
        assert!(p
            .filters
            .iter()
            .all(|f| matches!(&f.kind, FilterKind::Discovered(d) if d.decomposition == 0)));
        // Event filters cannot be merged or split.
        p.filters.push(analysed_event(FilterAction::Suppress, 1.0));
        assert!(p.merge(0, 2).is_err() && p.split(2).is_err());
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

        // Found sounds keep their patterns; unused patterns are not saved.
        p.add_discovery(discovery());
        p.decompositions.push(discovery().decomposition);
        p.save(&path).unwrap();
        let q = Project::load(&path).unwrap();
        assert_eq!(q.decompositions.len(), 1);
        assert_eq!(q.decompositions[0], discovery().decomposition);
        assert_eq!(q.filters.len(), 4);
        assert!(q.filters[2].suggested);
    }
}
