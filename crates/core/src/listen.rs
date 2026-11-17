//! Listening to one sound on its own, or to everything but it.
//!
//! Before a filter is trusted, the user wants to hear what it targets: the
//! picked sound *solo* (does the red really belong to the bird?) and the
//! recording *without* it (is anything of the bird left, did the rest
//! suffer?). Both are ordinary projects derived from the edited one, so
//! the engine, the bin sharing of [`crate::share`] and the display work
//! unchanged; only the actions and on/off states differ:
//!
//! | mode               | picked filter     | every other enabled filter |
//! |--------------------|-------------------|----------------------------|
//! | [`Listen::All`]    | as it is          | as it is                   |
//! | [`Listen::Solo`]   | on, **extract**   | **suppress**               |
//! | [`Listen::Without`]| on, **suppress**  | as it is                   |
//!
//! In solo, the other sounds are *suppressed* rather than switched off:
//! where they share bins with the picked sound, the bins' energy is split
//! between them (see [`crate::share`]), so only the picked sound's share is
//! heard; a noise reduction keeps the hiss out. Other extract filters
//! would otherwise let their sounds in through the union of extract masks.
//!
//! "Without" keeps everything else as the user set it up, so it is what
//! the export would sound like if the picked filter suppressed its sound.
//!
//! Because extract and suppress of the same mask are power complements,
//! with no other event filter overlapping the picked sound, solo and
//! without add up (in power) to the output with the picked filter off.

use crate::filter::{FilterAction, Project};
use serde::{Deserialize, Serialize};

/// What the user hears (and what the spectrogram draws in blue).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Listen {
    /// The output of all enabled filters.
    #[default]
    All,
    /// Only the picked sound.
    Solo,
    /// Everything but the picked sound.
    Without,
}

/// The sound [`Listen::Solo`] and [`Listen::Without`] refer to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// Filter number `.0` of the list.
    Filter(usize),
    /// The selection being edited ([`Project::preview`]), not a filter yet.
    Preview,
}

impl Project {
    /// The project to process so that `mode` is heard for `pick` (see the
    /// module documentation). Without a pick, or when the pick does not
    /// exist, the project is returned as it is.
    ///
    /// A picked preview becomes the last filter of the result (visible, so
    /// it is still drawn) and the result has no preview, so the selection
    /// is neither analysed nor searched for twice.
    pub fn for_listening(&self, mode: Listen, pick: Option<Pick>) -> Project {
        let mut p = self.clone();
        if mode == Listen::All {
            return p;
        }
        let index = match pick {
            Some(Pick::Filter(i)) if i < p.filters.len() => i,
            Some(Pick::Preview) => match p.preview.take() {
                Some(mut f) => {
                    f.visible = true;
                    p.filters.push(f);
                    p.filters.len() - 1
                }
                None => return p,
            },
            _ => return p,
        };
        for (i, f) in p.filters.iter_mut().enumerate() {
            if i == index {
                f.enabled = true;
                f.action = match mode {
                    Listen::Solo => FilterAction::Extract,
                    Listen::Without | Listen::All => FilterAction::Suppress,
                };
            } else if mode == Listen::Solo {
                f.action = FilterAction::Suppress;
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{RegionAnalysis, Selection};
    use crate::filter::{compute_gains, Filter, FilterKind};
    use crate::stft::StftConfig;
    use std::sync::Arc;

    /// An event filter whose analysis marks bin 1 of frames 0..4 with
    /// `value` (as the selection) and nothing else.
    fn marked(name: &str, action: FilterAction, value: f32) -> Filter {
        let mut f = Filter::event(
            name,
            Selection {
                start_s: 0.0,
                end_s: 0.1,
                low_hz: 0.0,
                high_hz: 1000.0,
            },
            action,
        );
        if let FilterKind::Event(e) = &mut f.kind {
            e.analysis = Some(Arc::new(RegionAnalysis {
                first_frame: 0,
                frames: 4,
                bins: 3,
                selected: vec![vec![0.0, value, 0.0]; 4],
                harmonics: vec![vec![0.0; 3]; 4],
                comodulated: vec![vec![0.0; 3]; 4],
                ..Default::default()
            }));
        }
        f
    }

    /// A project with a background noise reduction (index 0) and the given
    /// filters after it.
    fn project(filters: Vec<Filter>) -> Project {
        let mut p = Project {
            stft: StftConfig { fft_size: 4, hop: 2 }, // 3 bins, matching `marked`
            ..Project::default()
        };
        p.filters.extend(filters);
        p
    }

    /// Gain of every bin of frame 0 with power 1 everywhere.
    fn gains(p: &Project) -> Vec<f32> {
        let power = vec![vec![1.0f32; 3]; 4];
        compute_gains(p, &power, 0, &power, 0, &[], &[]).gain[0].clone()
    }

    #[test]
    fn all_and_missing_picks_leave_the_project_alone() {
        let p = project(vec![marked("bird", FilterAction::Suppress, 0.8)]);
        for (mode, pick) in [
            (Listen::All, Some(Pick::Filter(1))),
            (Listen::Solo, None),
            (Listen::Solo, Some(Pick::Filter(7))),
            (Listen::Without, Some(Pick::Preview)),
        ] {
            let q = p.for_listening(mode, pick);
            assert_eq!(q.filters.len(), 2);
            assert_eq!(q.filters[1].action, FilterAction::Suppress, "{mode:?} {pick:?}");
            assert!(q.filters[1].enabled);
        }
    }

    #[test]
    fn solo_extracts_the_pick_and_suppresses_all_others() {
        let mut off = marked("frog (off)", FilterAction::Extract, 0.5);
        off.enabled = false;
        let mut suggested = marked("bird", FilterAction::Suppress, 0.8);
        suggested.enabled = false;
        let p = project(vec![suggested, marked("owl", FilterAction::Extract, 0.3), off]);
        let q = p.for_listening(Listen::Solo, Some(Pick::Filter(1)));
        let state: Vec<_> = q.filters.iter().map(|f| (f.enabled, f.action)).collect();
        assert_eq!(
            state,
            [
                (true, FilterAction::Suppress),  // noise reduction keeps the hiss out
                (true, FilterAction::Extract),   // the pick, although it was off
                (true, FilterAction::Suppress),  // another extract filter
                (false, FilterAction::Suppress)  // off stays off
            ]
        );
        // The edited project itself is untouched.
        assert_eq!(
            (p.filters[1].enabled, p.filters[2].action),
            (false, FilterAction::Extract)
        );
    }

    #[test]
    fn without_suppresses_the_pick_and_keeps_the_rest() {
        let p = project(vec![
            marked("bird", FilterAction::Extract, 0.8),
            marked("owl", FilterAction::Extract, 0.3),
        ]);
        let q = p.for_listening(Listen::Without, Some(Pick::Filter(1)));
        let actions: Vec<_> = q.filters.iter().map(|f| f.action).collect();
        assert_eq!(
            actions,
            [FilterAction::Suppress, FilterAction::Suppress, FilterAction::Extract]
        );
    }

    #[test]
    fn a_picked_preview_becomes_a_drawn_filter() {
        let mut p = project(vec![]);
        let mut sel = marked("Selection", FilterAction::Suppress, 0.8);
        (sel.enabled, sel.visible) = (false, false);
        p.preview = Some(sel);
        let q = p.for_listening(Listen::Solo, Some(Pick::Preview));
        assert!(q.preview.is_none());
        let f = q.filters.last().expect("preview appended");
        assert_eq!((f.name.as_str(), f.enabled, f.visible), ("Selection", true, true));
        assert_eq!(f.action, FilterAction::Extract);
        assert!(q.shows(q.filters.len() - 1), "still drawn");
    }

    #[test]
    fn solo_and_without_add_up_to_the_output_without_the_pick() {
        // Without overlapping event filters, extract and suppress of the
        // same mask are power complements: |solo|² + |without|² equals the
        // output with the picked filter switched off, bin by bin.
        let p = project(vec![marked("bird", FilterAction::Suppress, 0.8)]);
        let solo = gains(&p.for_listening(Listen::Solo, Some(Pick::Filter(1))));
        let without = gains(&p.for_listening(Listen::Without, Some(Pick::Filter(1))));
        let mut off = p.clone();
        off.filters[1].enabled = false;
        let rest = gains(&off);
        for k in 0..3 {
            let sum = solo[k] * solo[k] + without[k] * without[k];
            assert!((sum - rest[k] * rest[k]).abs() < 1e-5, "bin {k}: {sum} vs {}", rest[k]);
        }
        // And the solo really is only the marked bin.
        assert!(solo[1] > 0.0);
        assert_eq!((solo[0], solo[2]), (0.0, 0.0));
    }

    #[test]
    fn solo_shares_bins_with_another_sound_instead_of_letting_it_in() {
        // Bird and owl claim the same bin. Soloing the bird must not play
        // the owl's share of it: with the owl suppressed, the bin is split,
        // so the bird's solo gain stays below the bird's own mask (0.8)
        // instead of the union an extracted owl would give.
        let p = project(vec![
            marked("bird", FilterAction::Extract, 0.8),
            marked("owl", FilterAction::Extract, 0.8),
        ]);
        let solo = gains(&p.for_listening(Listen::Solo, Some(Pick::Filter(1))));
        let both = gains(&p);
        assert!(solo[1] < both[1], "solo {} vs both {}", solo[1], both[1]);
    }
}
