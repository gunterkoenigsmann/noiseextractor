//! The detection settings of the selected object, as sliders: how strict
//! the analysis is about what belongs to the marked sound (harmonics,
//! common amplitude changes, channel level ratio and delay) and about what
//! counts as a repetition elsewhere.
//!
//! Each [`Param`] knows where its value lives in an
//! [`EventFilter`], its range, how it is shown, and whether changing it
//! needs a new analysis of the selection (the parts) or only new processing
//! (the repetitions). The sliders are integers `0..=STEPS`; [`Param::slider_pos`]
//! and [`Param::value_at`] convert. Free of GUI types, so it is tested.

use noiseextractor_core::filter::EventFilter;

/// Slider positions per parameter range.
pub const STEPS: i32 = 100;

/// Which group a parameter is shown in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// What else belongs to this occurrence of the sound.
    Parts,
    /// Where else the sound occurs.
    Repetitions,
}

/// One adjustable detection parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    /// Relative frequency tolerance around a predicted harmonic.
    HarmonicTolerance,
    /// Evidence a harmonic track needs to count.
    HarmonicThreshold,
    /// Envelope correlation a bin needs to count as co-modulated.
    ComodulationThreshold,
    /// Largest harmonic number searched.
    MaxHarmonic,
    /// Level ratio tolerance between channels for parts.
    PartLevelTolerance,
    /// Delay tolerance between channels for parts.
    PartDelayTolerance,
    /// Largest inter-channel delay searched (microphone spacing).
    MaxDelay,
    /// Shape correlation a repetition needs.
    MatchThreshold,
    /// Frequency shift allowed for a repetition.
    MaxShift,
    /// How much the direction counts for repetitions.
    SpatialWeight,
    /// Level ratio tolerance between channels for repetitions.
    RepetitionLevelTolerance,
    /// Delay tolerance between channels for repetitions.
    RepetitionDelayTolerance,
    /// Loudness above the noise a repetition of a pulsed sound needs.
    MinLevel,
}

impl Param {
    /// All parameters, in display order.
    pub const ALL: [Param; 13] = [
        Param::HarmonicTolerance,
        Param::HarmonicThreshold,
        Param::MaxHarmonic,
        Param::ComodulationThreshold,
        Param::PartLevelTolerance,
        Param::PartDelayTolerance,
        Param::MaxDelay,
        Param::MatchThreshold,
        Param::MaxShift,
        Param::MinLevel,
        Param::SpatialWeight,
        Param::RepetitionLevelTolerance,
        Param::RepetitionDelayTolerance,
    ];

    /// The group the slider is shown in.
    pub fn group(self) -> Group {
        match self {
            Param::MatchThreshold
            | Param::MaxShift
            | Param::MinLevel
            | Param::SpatialWeight
            | Param::RepetitionLevelTolerance
            | Param::RepetitionDelayTolerance => Group::Repetitions,
            _ => Group::Parts,
        }
    }

    /// Label next to the slider.
    pub fn label(self) -> &'static str {
        match self {
            Param::HarmonicTolerance => "Harmonic frequency tolerance",
            Param::HarmonicThreshold => "Harmonic evidence needed",
            Param::MaxHarmonic => "Highest harmonic",
            Param::ComodulationThreshold => "Amplitude-change correlation needed",
            Param::PartLevelTolerance => "Channel level tolerance",
            Param::PartDelayTolerance => "Channel delay tolerance",
            Param::MaxDelay => "Largest channel delay (mic spacing)",
            Param::MatchThreshold => "Shape match needed",
            Param::MaxShift => "Frequency shift allowed",
            Param::MinLevel => "Loudness needed (pulsed sounds)",
            Param::SpatialWeight => "Weight of the direction",
            Param::RepetitionLevelTolerance => "Channel level tolerance",
            Param::RepetitionDelayTolerance => "Channel delay tolerance",
        }
    }

    /// Tooltip explaining the effect.
    pub fn help(self) -> &'static str {
        match self {
            Param::HarmonicTolerance => {
                "How far (in % of its frequency) a partial may be from m/n times the tracked pitch"
            }
            Param::HarmonicThreshold => "Lower: more partials are taken as harmonics of the marked sound",
            Param::MaxHarmonic => "Highest multiple of the tracked pitch that is searched",
            Param::ComodulationThreshold => {
                "How closely a frequency's loudness must follow the marked part's to count as belonging to it"
            }
            Param::PartLevelTolerance => {
                "Stereo: a part whose channel level ratio differs by this much (dB) counts only half"
            }
            Param::PartDelayTolerance => {
                "Stereo: a part whose delay between channels differs by this much (samples) counts only half"
            }
            Param::MaxDelay => "Stereo: largest delay between the channels searched, in samples",
            Param::MatchThreshold => "How similar in shape another place must be to count as a repetition",
            Param::MaxShift => "How much higher or lower (in FFT bins) a repetition may be",
            Param::MinLevel => {
                "Pulsed sounds (insects, trills): how far (dB) a repetition must be above the noise; \
                 lower finds more distant ones"
            }
            Param::SpatialWeight => "Stereo: 0 ignores where a repetition comes from, 1 requires the same place",
            Param::RepetitionLevelTolerance => {
                "Stereo: level ratio difference (dB) at which a repetition counts as coming from elsewhere"
            }
            Param::RepetitionDelayTolerance => {
                "Stereo: delay difference (samples) at which a repetition counts as coming from elsewhere"
            }
        }
    }

    /// Range of the value, `(min, max)`.
    fn range(self) -> (f32, f32) {
        match self {
            Param::HarmonicTolerance => (0.005, 0.1),
            Param::HarmonicThreshold => (0.1, 0.95),
            Param::MaxHarmonic => (2.0, 24.0),
            Param::ComodulationThreshold => (0.3, 0.99),
            Param::PartLevelTolerance | Param::RepetitionLevelTolerance => (0.5, 12.0),
            Param::PartDelayTolerance | Param::RepetitionDelayTolerance => (0.25, 16.0),
            Param::MaxDelay => (1.0, 256.0),
            Param::MatchThreshold => (0.2, 0.95),
            Param::MaxShift => (0.0, 8.0),
            Param::MinLevel => (0.0, 30.0),
            Param::SpatialWeight => (0.0, 1.0),
        }
    }

    /// Whether the value is a whole number.
    fn integral(self) -> bool {
        matches!(self, Param::MaxHarmonic | Param::MaxShift)
    }

    /// Whether the parameter only matters for multi-channel recordings.
    pub fn spatial(self) -> bool {
        matches!(
            self,
            Param::PartLevelTolerance
                | Param::PartDelayTolerance
                | Param::MaxDelay
                | Param::SpatialWeight
                | Param::RepetitionLevelTolerance
                | Param::RepetitionDelayTolerance
        )
    }

    /// Whether a change needs a new analysis of the selection (the parts
    /// and the template), not only new processing.
    pub fn reanalyse(self) -> bool {
        self.group() == Group::Parts
    }

    /// The current value in `e`.
    pub fn get(self, e: &EventFilter) -> f32 {
        let (a, s) = (&e.params, &e.similar);
        match self {
            Param::HarmonicTolerance => a.harmonic_tolerance,
            Param::HarmonicThreshold => a.harmonic_threshold,
            Param::MaxHarmonic => a.max_harmonic as f32,
            Param::ComodulationThreshold => a.comodulation_threshold,
            Param::PartLevelTolerance => a.spatial_level_tolerance_db,
            Param::PartDelayTolerance => a.spatial_delay_tolerance_samples,
            Param::MaxDelay => a.max_delay_samples,
            Param::MatchThreshold => s.threshold,
            Param::MaxShift => s.max_shift_bins as f32,
            Param::MinLevel => s.min_level_db,
            Param::SpatialWeight => s.spatial_weight,
            Param::RepetitionLevelTolerance => s.level_tolerance_db,
            Param::RepetitionDelayTolerance => s.delay_tolerance_samples,
        }
    }

    /// Sets the value in `e` (clamped to the range, rounded if integral).
    pub fn set(self, e: &mut EventFilter, v: f32) {
        let (lo, hi) = self.range();
        let v = v.clamp(lo, hi);
        let (a, s) = (&mut e.params, &mut e.similar);
        match self {
            Param::HarmonicTolerance => a.harmonic_tolerance = v,
            Param::HarmonicThreshold => a.harmonic_threshold = v,
            Param::MaxHarmonic => a.max_harmonic = v.round() as u32,
            Param::ComodulationThreshold => a.comodulation_threshold = v,
            Param::PartLevelTolerance => a.spatial_level_tolerance_db = v,
            Param::PartDelayTolerance => a.spatial_delay_tolerance_samples = v,
            Param::MaxDelay => a.max_delay_samples = v,
            Param::MatchThreshold => s.threshold = v,
            Param::MaxShift => s.max_shift_bins = v.round() as usize,
            Param::MinLevel => s.min_level_db = v,
            Param::SpatialWeight => s.spatial_weight = v,
            Param::RepetitionLevelTolerance => s.level_tolerance_db = v,
            Param::RepetitionDelayTolerance => s.delay_tolerance_samples = v,
        }
    }

    /// Number of slider positions minus one: one per whole number for
    /// integral parameters, [`STEPS`] otherwise.
    pub fn steps(self) -> i32 {
        let (lo, hi) = self.range();
        if self.integral() {
            (hi - lo).round() as i32
        } else {
            STEPS
        }
    }

    /// Slider position of value `v`.
    pub fn slider_pos(self, v: f32) -> i32 {
        let (lo, hi) = self.range();
        (((v.clamp(lo, hi) - lo) / (hi - lo)) * self.steps() as f32).round() as i32
    }

    /// Value at slider position `i`.
    pub fn value_at(self, i: i32) -> f32 {
        let (lo, hi) = self.range();
        let v = lo + (hi - lo) * i.clamp(0, self.steps()) as f32 / self.steps() as f32;
        if self.integral() {
            v.round()
        } else {
            v
        }
    }

    /// The value as shown next to the slider, with its unit.
    pub fn format(self, v: f32) -> String {
        match self {
            Param::HarmonicTolerance => format!("{:.1} %", v * 100.0),
            Param::MaxHarmonic => format!("{v:.0}"),
            Param::MaxShift => format!("± {v:.0} bins"),
            Param::PartLevelTolerance | Param::RepetitionLevelTolerance | Param::MinLevel => format!("{v:.1} dB"),
            Param::PartDelayTolerance | Param::RepetitionDelayTolerance | Param::MaxDelay => {
                format!("{v:.2} samples")
            }
            _ => format!("{v:.2}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noiseextractor_core::analysis::Selection;

    /// An event filter with default settings.
    fn filter() -> EventFilter {
        EventFilter::new(Selection {
            start_s: 0.0,
            end_s: 1.0,
            low_hz: 100.0,
            high_hz: 200.0,
            outline: Vec::new(),
        })
    }

    #[test]
    fn defaults_lie_inside_the_ranges_and_survive_the_slider() {
        let e = filter();
        for p in Param::ALL {
            let v = p.get(&e);
            let (lo, hi) = p.range();
            assert!((lo..=hi).contains(&v), "{p:?} default {v} outside {lo}..{hi}");
            // Going through the slider moves a value by at most half a step.
            let back = p.value_at(p.slider_pos(v));
            assert!(
                (back - v).abs() <= (hi - lo) / p.steps() as f32 / 2.0 + 1e-6,
                "{p:?}: {v} -> {back}"
            );
        }
    }

    #[test]
    fn set_writes_the_right_field_and_clamps() {
        for p in Param::ALL {
            let mut e = filter();
            let (lo, hi) = p.range();
            p.set(&mut e, hi + 100.0);
            assert_eq!(p.get(&e), hi, "{p:?}");
            p.set(&mut e, lo - 100.0);
            assert_eq!(p.get(&e), lo, "{p:?}");
            // No other parameter changed.
            let fresh = filter();
            for q in Param::ALL.into_iter().filter(|&q| q != p) {
                assert_eq!(q.get(&e), q.get(&fresh), "{p:?} changed {q:?}");
            }
        }
    }

    #[test]
    fn integral_parameters_step_by_one() {
        assert_eq!(Param::MaxHarmonic.steps(), 22);
        assert_eq!(Param::MaxHarmonic.value_at(3), 5.0);
        assert_eq!(Param::MaxShift.slider_pos(2.0), 2);
        assert_eq!(Param::SpatialWeight.steps(), STEPS);
    }

    #[test]
    fn groups_decide_what_is_recomputed() {
        assert!(Param::ComodulationThreshold.reanalyse());
        assert!(Param::MaxDelay.reanalyse());
        assert!(!Param::SpatialWeight.reanalyse());
        assert!(Param::SpatialWeight.spatial() && !Param::MatchThreshold.spatial());
        assert!(!Param::MinLevel.reanalyse() && !Param::MinLevel.spatial());
        assert_eq!(Param::MinLevel.format(3.0), "3.0 dB");
        assert_eq!(Param::HarmonicTolerance.format(0.03), "3.0 %");
    }
}
