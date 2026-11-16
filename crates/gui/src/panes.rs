//! The main window's dockable panes, arranged by wxWidgets' AUI manager
//! (as in losslessaudiocut): the spectrogram in the centre, the filter list
//! and the detection settings on the right, the waveform overview on top,
//! the log at the bottom. The user can drag panes to another edge, float,
//! resize and close them; the View menu shows and hides each one.
//!
//! The arrangement (AUI "perspective") and the window size are stored in
//! wxWidgets' per-user config store when the window closes and restored at
//! the next start.

use wxdragon::config::{Config, ConfigStyle};
use wxdragon::prelude::*;
use wxdragon::widgets::aui_manager::{AuiManager, PaneInfo};

/// Key of the stored AUI perspective.
const PERSPECTIVE: &str = "/Layout/Perspective";
/// Key of the stored window width.
const WIDTH: &str = "/Layout/Width";
/// Key of the stored window height.
const HEIGHT: &str = "/Layout/Height";
/// Key of the layout version the perspective was stored with.
const VERSION: &str = "/Layout/Version";
/// Bumped when panes are added or renamed, so an old stored perspective
/// (which would hide the new panes) is ignored.
const LAYOUT_VERSION: i64 = 1;

/// Name of the centre pane (the spectrogram). It cannot be closed, so it is
/// no [`Pane`].
const SPECTROGRAM: &str = "spectrogram";

/// First View menu id of the pane toggles; one per [`Pane::ALL`] entry.
const ID_FIRST_PANE: i32 = ID_HIGHEST + 100;

/// A pane that can be shown and hidden.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pane {
    /// The filter list and the selected filter's details.
    Filters,
    /// Sliders for the selected object's detection settings.
    Detection,
    /// The whole recording's waveform with the visible part framed.
    Overview,
    /// Every status message with its time.
    Log,
}

impl Pane {
    /// All panes, in View menu order.
    pub const ALL: [Pane; 4] = [Pane::Filters, Pane::Detection, Pane::Overview, Pane::Log];

    /// Stable name in the stored perspective.
    pub fn name(self) -> &'static str {
        match self {
            Pane::Filters => "filters",
            Pane::Detection => "detection",
            Pane::Overview => "overview",
            Pane::Log => "log",
        }
    }

    /// Title bar of the pane.
    pub fn caption(self) -> &'static str {
        match self {
            Pane::Filters => "Filters",
            Pane::Detection => "Detection settings",
            Pane::Overview => "Overview",
            Pane::Log => "Log",
        }
    }

    /// View menu entry.
    pub fn menu_label(self) -> &'static str {
        match self {
            Pane::Filters => "&Filters",
            Pane::Detection => "&Detection settings",
            Pane::Overview => "Waveform &overview",
            Pane::Log => "Lo&g",
        }
    }

    /// Status bar help of the View menu entry.
    pub fn menu_help(self) -> &'static str {
        match self {
            Pane::Filters => "Show the filter list",
            Pane::Detection => "Show how strictly parts and repetitions of the selected sound are detected",
            Pane::Overview => "Show the whole recording, with the part in the spectrogram framed",
            Pane::Log => "Show every status message with its time",
        }
    }

    /// The View menu command that shows or hides the pane.
    pub fn menu_id(self) -> i32 {
        let i = Pane::ALL.iter().position(|&p| p == self).unwrap_or(0);
        ID_FIRST_PANE + i as i32
    }

    /// The pane that View menu command `id` toggles.
    pub fn for_menu_id(id: i32) -> Option<Pane> {
        let i = usize::try_from(id - ID_FIRST_PANE).ok()?;
        Pane::ALL.get(i).copied()
    }
}

/// The per-user config store.
fn config() -> Config {
    Config::new("noiseextractor", None, None, None, ConfigStyle::USE_LOCAL_FILE)
}

/// The stored window size, or a default.
pub fn stored_size() -> Size {
    let c = config();
    let w = c.read_long(WIDTH, 1280).clamp(400, 10_000) as i32;
    let h = c.read_long(HEIGHT, 800).clamp(300, 10_000) as i32;
    Size::new(w, h)
}

/// The windows the panes hold.
pub struct Contents<'a> {
    /// The spectrogram, in the centre.
    pub spectrogram: &'a Panel,
    /// The filter list, on the right.
    pub filters: &'a ScrolledWindow,
    /// The detection sliders, right, under the filters.
    pub detection: &'a ScrolledWindow,
    /// The waveform overview, on top.
    pub overview: &'a Panel,
    /// The log, at the bottom.
    pub log: &'a TextCtrl,
}

/// The AUI manager of the main window and the frame it manages. Copy, so
/// event handlers can hold it.
#[derive(Clone, Copy)]
pub struct Panes {
    /// Arranges the panes.
    aui: AuiManager,
    /// The main window.
    frame: Frame,
}

impl Panes {
    /// Docks the contents around the spectrogram and restores the stored
    /// arrangement.
    pub fn new(frame: &Frame, c: &Contents) -> Panes {
        let aui = AuiManager::builder(frame).build();
        aui.add_pane_with_info(
            c.spectrogram,
            PaneInfo::new().with_name(SPECTROGRAM).center_pane().min_size(300, 150),
        );
        let side = |p: Pane| {
            PaneInfo::new()
                .with_name(p.name())
                .with_caption(p.caption())
                .close_button(true)
        };
        aui.add_pane_with_info(
            c.filters,
            side(Pane::Filters).right().best_size(330, 420).min_size(220, 200),
        );
        aui.add_pane_with_info(
            c.detection,
            side(Pane::Detection)
                .right()
                .position(1)
                .best_size(330, 420)
                .min_size(220, 120),
        );
        aui.add_pane_with_info(
            c.overview,
            side(Pane::Overview).top().best_size(600, 70).min_size(100, 30),
        );
        aui.add_pane_with_info(
            c.log,
            side(Pane::Log)
                .bottom()
                .best_size(600, 110)
                .min_size(150, 50)
                .maximize_button(true)
                .hide(),
        );
        let panes = Panes { aui, frame: *frame };
        let cfg = config();
        let stored = cfg.read_string(PERSPECTIVE, "");
        if !stored.is_empty() && cfg.read_long(VERSION, 0) == LAYOUT_VERSION {
            aui.load_perspective(&stored, false);
        }
        aui.update();
        // Keep the View menu's check marks right when the user closes a
        // pane with its own close button.
        frame.on_menu_opened(move |e| {
            panes.sync_menu();
            e.skip(true);
        });
        panes
    }

    /// Whether `pane` is shown.
    pub fn is_shown(&self, pane: Pane) -> bool {
        self.aui.is_pane_shown(pane.name())
    }

    /// Shows `pane` if hidden, else hides it.
    pub fn toggle(&self, pane: Pane) {
        self.aui.set_pane_shown(pane.name(), !self.is_shown(pane));
        self.aui.update();
        self.sync_menu();
    }

    /// Sets the View menu's check marks from what is shown.
    pub fn sync_menu(&self) {
        let Some(bar) = self.frame.get_menu_bar() else {
            return;
        };
        for pane in Pane::ALL {
            bar.check_item(pane.menu_id(), self.is_shown(pane));
        }
    }

    /// Stores the arrangement and the window size, for the next start.
    /// Failing to store them only means they are not remembered.
    pub fn save(&self) {
        let c = config();
        c.write_string(PERSPECTIVE, &self.aui.save_perspective());
        c.write_long(VERSION, LAYOUT_VERSION);
        if !self.frame.is_maximized() && !self.frame.is_iconized() {
            let size = self.frame.get_size();
            c.write_long(WIDTH, i64::from(size.width));
            c.write_long(HEIGHT, i64::from(size.height));
        }
        c.flush(false);
    }

    /// Detaches the panes before the frame goes away, as wxWidgets asks.
    pub fn uninit(&self) {
        self.aui.uninit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pane_has_its_own_menu_command_and_name() {
        for pane in Pane::ALL {
            assert_eq!(Pane::for_menu_id(pane.menu_id()), Some(pane));
            assert_ne!(pane.name(), SPECTROGRAM);
            assert!(!pane.caption().is_empty() && !pane.menu_help().is_empty());
        }
        assert_eq!(Pane::for_menu_id(ID_FIRST_PANE - 1), None);
        assert_eq!(Pane::for_menu_id(ID_FIRST_PANE + Pane::ALL.len() as i32), None);
    }
}
