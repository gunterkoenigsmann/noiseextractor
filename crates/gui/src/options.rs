//! Settings remembered between runs, kept in wxWidgets' per-user config
//! store (`panes::config`). The settings dialog (`settings.rs`) edits
//! them; this module holds the rules (ranges, defaults, how they are
//! stored) free of widgets, so they can be unit-tested.
//!
//! Only things a user sets once and then rarely touches belong here. What
//! belongs to one filter (its name, parts, maximum reduction) is saved in
//! the project; the window layout is `panes.rs`'s business.

use std::ops::RangeInclusive;

use noiseextractor_core::discover::DiscoverParams;
use noiseextractor_core::normalize::{DEFAULT_PEAK_DBFS, PEAK_DBFS_RANGE};

use crate::panes::config;

/// Key of [`Options::db_floor`].
const DB_FLOOR: &str = "/Spectrogram/FloorDb";
/// Key of [`Options::db_ceiling`].
const DB_CEILING: &str = "/Spectrogram/CeilingDb";
/// Key of [`Options::patterns`].
const PATTERNS: &str = "/FindSounds/Patterns";
/// Key of [`Options::group`].
const GROUP: &str = "/FindSounds/Group";
/// Key of [`Options::normalize`].
const NORMALIZE: &str = "/Export/Normalize";
/// Key of [`Options::peak_dbfs_tenths`].
const PEAK: &str = "/Export/PeakDbfsTenths";

/// Darkest level the spectrogram's floor can be set to, and the brightest
/// its ceiling can be, in dB (the scale of `render::View`).
pub const DB_RANGE: RangeInclusive<i32> = -160..=20;
/// Narrowest span between floor and ceiling, in dB: below that the
/// spectrogram is little more than black and white.
pub const MIN_DB_SPAN: i32 = 20;
/// Pattern counts "Find sounds" offers.
pub const PATTERN_RANGE: RangeInclusive<u32> = 1..=16;

/// The remembered settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Spectrogram: level drawn black, in dB.
    pub db_floor: i32,
    /// Spectrogram: level drawn at full colour, in dB.
    pub db_ceiling: i32,
    /// Find sounds: how many spectral patterns to look for.
    pub patterns: u32,
    /// Find sounds: suggest patterns that come and go together (and from
    /// the same direction) as one sound.
    pub group: bool,
    /// Export: scale the output so its loudest sample reaches
    /// [`Options::peak_dbfs`].
    pub normalize: bool,
    /// Export: the normalization target in tenths of a dBFS. Integer, so
    /// the struct is `Eq` and the stored value is exact.
    pub peak_dbfs_tenths: i32,
}

impl Default for Options {
    fn default() -> Self {
        let d = DiscoverParams::default();
        Options {
            // What the spectrogram used before these were settings.
            db_floor: -100,
            db_ceiling: -10,
            patterns: d.components as u32,
            group: d.group,
            normalize: false,
            peak_dbfs_tenths: tenths(DEFAULT_PEAK_DBFS),
        }
    }
}

impl Options {
    /// The options with every value pulled into its range, and the
    /// spectrogram's ceiling at least [`MIN_DB_SPAN`] above its floor (the
    /// floor gives way, so the one just set by the user wins only if it is
    /// the ceiling; [`Options::with_floor`] handles the other case).
    pub fn normalized(mut self) -> Self {
        let (lo, hi) = (*DB_RANGE.start(), *DB_RANGE.end());
        self.db_ceiling = self.db_ceiling.clamp(lo + MIN_DB_SPAN, hi);
        self.db_floor = self.db_floor.clamp(lo, self.db_ceiling - MIN_DB_SPAN);
        self.patterns = self.patterns.clamp(*PATTERN_RANGE.start(), *PATTERN_RANGE.end());
        self.peak_dbfs_tenths = self
            .peak_dbfs_tenths
            .clamp(tenths(PEAK_DBFS_RANGE.0), tenths(PEAK_DBFS_RANGE.1));
        self
    }

    /// Sets the floor, pushing the ceiling up if they would be closer than
    /// [`MIN_DB_SPAN`] (and the floor down where the ceiling cannot go).
    pub fn with_floor(mut self, floor: i32) -> Self {
        let (lo, hi) = (*DB_RANGE.start(), *DB_RANGE.end());
        self.db_floor = floor.clamp(lo, hi - MIN_DB_SPAN);
        self.db_ceiling = self.db_ceiling.max(self.db_floor + MIN_DB_SPAN);
        self.normalized()
    }

    /// Sets the ceiling, pushing the floor down if needed.
    pub fn with_ceiling(mut self, ceiling: i32) -> Self {
        self.db_ceiling = ceiling;
        self.normalized()
    }

    /// The normalization target in dBFS.
    pub fn peak_dbfs(&self) -> f32 {
        self.peak_dbfs_tenths as f32 / 10.0
    }

    /// The normalization target to export with, `None` when it is off.
    pub fn normalize_target(&self) -> Option<f32> {
        self.normalize.then(|| self.peak_dbfs())
    }

    /// "Find sounds" parameters for these options (the search range and
    /// the expert parameters stay at their defaults).
    pub fn discover_params(&self) -> DiscoverParams {
        DiscoverParams {
            components: self.patterns as usize,
            group: self.group,
            ..DiscoverParams::default()
        }
    }

    /// The stored options; defaults for whatever is not stored.
    pub fn load() -> Self {
        let c = config();
        let d = Options::default();
        let int = |key: &str, default: i32| clamp_i32(c.read_long(key, default.into()));
        Options {
            db_floor: int(DB_FLOOR, d.db_floor),
            db_ceiling: int(DB_CEILING, d.db_ceiling),
            patterns: u32::try_from(int(PATTERNS, d.patterns as i32)).unwrap_or(d.patterns),
            group: c.read_bool(GROUP, d.group),
            normalize: c.read_bool(NORMALIZE, d.normalize),
            peak_dbfs_tenths: int(PEAK, d.peak_dbfs_tenths),
        }
        .normalized()
    }

    /// Stores the options. Failing to store them only means they are not
    /// remembered, which is not worth bothering the user with.
    pub fn save(&self) {
        let c = config();
        c.write_long(DB_FLOOR, self.db_floor.into());
        c.write_long(DB_CEILING, self.db_ceiling.into());
        c.write_long(PATTERNS, self.patterns.into());
        c.write_bool(GROUP, self.group);
        c.write_bool(NORMALIZE, self.normalize);
        c.write_long(PEAK, self.peak_dbfs_tenths.into());
        c.flush(false);
    }
}

/// `db` in tenths of a dB, rounded.
pub fn tenths(db: f32) -> i32 {
    (db * 10.0).round() as i32
}

/// A stored number pulled into `i32` (a hand-edited store may hold
/// anything); [`Options::normalized`] does the rest.
fn clamp_i32(stored: i64) -> i32 {
    i32::try_from(stored.clamp(i32::MIN.into(), i32::MAX.into())).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_keep_what_the_program_did_before() {
        let o = Options::default();
        assert_eq!((o.db_floor, o.db_ceiling), (-100, -10));
        assert_eq!(o.discover_params(), DiscoverParams::default());
        // Normalizing is a choice per use, so it starts off.
        assert_eq!(o.normalize_target(), None);
        assert_eq!(o.peak_dbfs(), -1.0);
        assert_eq!(o, o.normalized(), "defaults are within the rules");
    }

    #[test]
    fn out_of_range_values_are_pulled_in() {
        let o = Options {
            db_floor: -1000,
            db_ceiling: 1000,
            patterns: 0,
            group: true,
            normalize: true,
            peak_dbfs_tenths: 60,
        }
        .normalized();
        assert_eq!((o.db_floor, o.db_ceiling), (-160, 20));
        assert_eq!(o.patterns, 1);
        assert_eq!(o.normalize_target(), Some(0.0), "never above full scale");
        let o = Options {
            patterns: 99,
            peak_dbfs_tenths: -9999,
            ..o
        }
        .normalized();
        assert_eq!(o.patterns, 16);
        assert_eq!(o.peak_dbfs(), -40.0);
        assert_eq!(clamp_i32(i64::MAX), i32::MAX);
        assert_eq!(clamp_i32(-7), -7);
    }

    #[test]
    fn floor_and_ceiling_keep_their_distance() {
        let o = Options::default();
        // Raising the floor past the ceiling lifts the ceiling ...
        let up = o.with_floor(-15);
        assert_eq!((up.db_floor, up.db_ceiling), (-15, 5));
        // ... as far as it goes; then the floor stops.
        let top = o.with_floor(10);
        assert_eq!((top.db_floor, top.db_ceiling), (0, 20));
        // Lowering the ceiling below the floor pushes the floor down.
        let down = o.with_ceiling(-110);
        assert_eq!((down.db_floor, down.db_ceiling), (-130, -110));
        let bottom = o.with_ceiling(-500);
        assert_eq!((bottom.db_floor, bottom.db_ceiling), (-160, -140));
    }

    #[test]
    fn find_sounds_settings_reach_the_search() {
        let o = Options {
            patterns: 9,
            group: false,
            ..Options::default()
        };
        let p = o.discover_params();
        assert_eq!((p.components, p.group, p.range_s), (9, false, None));
        assert_eq!(tenths(-1.04), -10);
        assert_eq!(tenths(-0.25), -3);
    }
}
