//! The main window's toolbar: one button per frequently used menu command.
//!
//! Every tool reuses the command id of a menu item, so a click arrives as
//! the same `EventType::MENU` event as the menu item and is handled in one
//! place (`bind_events` in `main.rs`). The toolbar therefore holds no logic
//! of its own; this module only describes the buttons ([`ITEMS`]) and draws
//! their icons.
//!
//! The icons are small inline SVGs rendered by wxWidgets (nanosvg) through a
//! [`BitmapBundle`], so they stay sharp on HiDPI screens and need no image
//! files in the packages. wxWidgets' stock art has no play, zoom or
//! spectrogram symbols, and mixing stock and own icons would look patchy.

use wxdragon::prelude::*;

/// Design size of the icons in device-independent pixels; the SVGs use a
/// matching `0 0 24 24` view box.
const ICON_SIZE: i32 = 24;

/// One entry of the toolbar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    /// A button that sends the menu command `id`.
    Tool {
        /// Menu command id, shared with the menu item that does the same.
        id: i32,
        /// Short label (shown under the icon if text labels are enabled).
        label: &'static str,
        /// Tooltip, naming the keyboard shortcut where there is one.
        help: &'static str,
        /// What to draw.
        icon: Icon,
    },
    /// One of a group of mutually exclusive buttons (adjacent radio items
    /// form a group; any other item ends it); shows which mode is on. The
    /// state is set from the program with `ToolBar::toggle_tool`, see
    /// `set_listen` and `set_select_tool` in `main.rs`.
    Radio {
        /// Menu command id, shared with the radio menu item.
        id: i32,
        /// Short label.
        label: &'static str,
        /// Tooltip, naming the keyboard shortcut.
        help: &'static str,
        /// What to draw.
        icon: Icon,
    },
    /// A divider between groups of tools.
    Separator,
}

/// The toolbar's contents, left to right: files, playback, marking and
/// building filters, view.
pub const ITEMS: &[Item] = &[
    Item::Tool {
        id: crate::ID_OPEN_AUDIO,
        label: "Open",
        help: "Open a recording (Ctrl+O)",
        icon: Icon::OpenAudio,
    },
    Item::Tool {
        id: crate::ID_OPEN_PROJECT,
        label: "Open project",
        help: "Open a filter project",
        icon: Icon::OpenProject,
    },
    Item::Tool {
        id: crate::ID_SAVE_PROJECT,
        label: "Save",
        help: "Save the filter project (Ctrl+S)",
        icon: Icon::Save,
    },
    Item::Tool {
        id: crate::ID_EXPORT,
        label: "Export",
        help: "Export the processed audio as WAV (Ctrl+E)",
        icon: Icon::Export,
    },
    Item::Separator,
    Item::Tool {
        id: crate::ID_PLAY,
        label: "Play",
        help: "Play / pause the selection or from the play head (Space, Ctrl+P)",
        icon: Icon::PlayPause,
    },
    Item::Radio {
        id: crate::ID_HEAR_ALL,
        label: "Hear all",
        help: "Hear everything (Ctrl+1)",
        icon: Icon::HearAll,
    },
    Item::Radio {
        id: crate::ID_HEAR_SOLO,
        label: "Hear sound",
        help: "Hear only the picked sound: the selection, else the filter picked in the list (Ctrl+2)",
        icon: Icon::HearSolo,
    },
    Item::Radio {
        id: crate::ID_HEAR_WITHOUT,
        label: "Hear without",
        help: "Hear everything but the picked sound (Ctrl+3)",
        icon: Icon::HearWithout,
    },
    Item::Separator,
    Item::Radio {
        id: crate::ID_SELECT_RECT,
        label: "Rectangle",
        help: "Drag a rectangle to mark a sound (Ctrl+R)",
        icon: Icon::SelectRect,
    },
    Item::Radio {
        id: crate::ID_SELECT_FREE,
        label: "Free-form",
        help: "Draw around a sound to mark it, for shapes a rectangle does not fit (Ctrl+D)",
        icon: Icon::SelectFree,
    },
    Item::Tool {
        id: crate::ID_ADD_SUPPRESS,
        label: "Suppress",
        help: "Add a filter that suppresses the selected sound",
        icon: Icon::Suppress,
    },
    Item::Tool {
        id: crate::ID_ADD_EXTRACT,
        label: "Extract",
        help: "Add a filter that keeps only the selected sound",
        icon: Icon::Extract,
    },
    Item::Tool {
        id: crate::ID_LEARN_NOISE,
        label: "Learn noise",
        help: "Learn the background noise from the selection",
        icon: Icon::LearnNoise,
    },
    Item::Tool {
        id: crate::ID_FIND_SOUNDS,
        label: "Find sounds",
        help: "Find the recording's sounds, in the selection if there is one (Ctrl+F)",
        icon: Icon::FindSounds,
    },
    Item::Separator,
    Item::Tool {
        id: crate::ID_ZOOM_FIT,
        label: "Whole recording",
        help: "Show the whole recording (Ctrl+0)",
        icon: Icon::ZoomFit,
    },
    Item::Tool {
        id: crate::ID_LOG_FREQ,
        label: "Log. axis",
        help: "Toggle linear / logarithmic frequency axis (Ctrl+L)",
        icon: Icon::LogFreq,
    },
];

/// Colours an icon is drawn with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Palette {
    /// Outlines and symbols; must contrast with the toolbar background.
    pub fg: &'static str,
    /// The user's selection; green as in the spectrogram overlay.
    pub selection: &'static str,
    /// Removed or found sound; red as in the spectrogram overlay.
    pub found: &'static str,
}

impl Palette {
    /// For light toolbar backgrounds.
    pub const LIGHT: Palette = Palette {
        fg: "#303030",
        selection: "#1e8c32",
        found: "#c8281e",
    };
    /// For dark toolbar backgrounds: lighter strokes, brighter accents.
    pub const DARK: Palette = Palette {
        fg: "#e0e0e0",
        selection: "#4cd264",
        found: "#ff5a4b",
    };

    /// The palette that matches the system's light or dark appearance.
    pub fn current() -> Palette {
        let dark = wxdragon::appearance::get_system_appearance().is_some_and(|a| a.is_using_dark_background());
        if dark {
            Palette::DARK
        } else {
            Palette::LIGHT
        }
    }
}

/// The icon of a tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Icon {
    /// A folder with a waveform.
    OpenAudio,
    /// A folder with list lines (a filter list).
    OpenProject,
    /// A floppy disk.
    Save,
    /// A waveform with an arrow leaving it.
    Export,
    /// Play triangle next to pause bars.
    PlayPause,
    /// A speaker sending out sound waves: everything.
    HearAll,
    /// A speaker with only the (green) picked sound.
    HearSolo,
    /// A speaker with the (red) picked sound struck out.
    HearWithout,
    /// A dashed (green) rectangle with the pointer at a corner.
    SelectRect,
    /// A dashed (green) lasso loop around a U-shaped sweep.
    SelectFree,
    /// The (green) selection with a (red) minus.
    Suppress,
    /// The (green) selection with an arrow out of it.
    Extract,
    /// A noise floor with the (red) threshold learned from it.
    LearnNoise,
    /// A magnifier over a spectral blob.
    FindSounds,
    /// Corners pointing outwards.
    ZoomFit,
    /// Axes with a logarithmic curve.
    LogFreq,
}

impl Icon {
    /// All icons, for tests.
    #[cfg(test)]
    const ALL: [Icon; 16] = [
        Icon::OpenAudio,
        Icon::OpenProject,
        Icon::Save,
        Icon::Export,
        Icon::PlayPause,
        Icon::HearAll,
        Icon::HearSolo,
        Icon::HearWithout,
        Icon::SelectRect,
        Icon::SelectFree,
        Icon::Suppress,
        Icon::Extract,
        Icon::LearnNoise,
        Icon::FindSounds,
        Icon::ZoomFit,
        Icon::LogFreq,
    ];

    /// The SVG elements of the icon (without the `<svg>` wrapper), drawn on
    /// a 24×24 grid with the colours of `p`.
    fn body(self, p: &Palette) -> String {
        let fg = p.fg;
        let sel = p.selection;
        let found = p.found;
        // Outline style shared by most shapes.
        let line =
            format!(r#"fill="none" stroke="{fg}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round""#);
        // A folder; both "open" icons use it.
        let folder = format!(r#"<path d="M2 6h7l2 2h11v12H2z" {line}/>"#);
        // A small speaker on the left; the "hear" icons use it.
        let speaker = format!(r#"<path d="M2 9h3l4-4v14l-4-4H2z" fill="{fg}"/>"#);
        match self {
            Icon::OpenAudio => format!(
                r#"{folder}<path d="M5 14l2-3 2 5 2-6 2 7 2-4 2 1" fill="none" stroke="{sel}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>"#
            ),
            Icon::OpenProject => format!(
                r#"{folder}<path d="M6 12h12M6 15h12M6 18h8" fill="none" stroke="{sel}" stroke-width="1.6" stroke-linecap="round"/>"#
            ),
            Icon::Save => format!(
                r#"<path d="M3 3h15l3 3v15H3z" {line}/><path d="M7 3v6h9V3" {line}/><rect x="7" y="14" width="10" height="7" fill="{fg}"/>"#
            ),
            Icon::Export => format!(
                r#"<path d="M2 12l2-5 2 10 2-8 2 6 2-3" {line}/><path d="M13 12h9M18 8l4 4-4 4" fill="none" stroke="{sel}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>"#
            ),
            Icon::PlayPause => format!(
                r#"<path d="M3 4l10 8-10 8z" fill="{fg}"/><rect x="15" y="5" width="2.5" height="14" fill="{fg}"/><rect x="19.5" y="5" width="2.5" height="14" fill="{fg}"/>"#
            ),
            Icon::HearAll => {
                format!(r#"{speaker}<path d="M12 9a4 4 0 0 1 0 6M15 6a8 8 0 0 1 0 12M18 3a12 12 0 0 1 0 18" {line}/>"#)
            }
            Icon::HearSolo => format!(
                r#"{speaker}<path d="M12 12l2-4 2 8 2-6 2 4 2-2" fill="none" stroke="{sel}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>"#
            ),
            Icon::HearWithout => format!(
                r#"{speaker}<path d="M12 9a4 4 0 0 1 0 6M15 6a8 8 0 0 1 0 12" {line}/><path d="M13 20L23 4" stroke="{found}" stroke-width="2.5" stroke-linecap="round"/>"#
            ),
            Icon::SelectRect => format!(
                r#"<rect x="2" y="4" width="16" height="13" fill="none" stroke="{sel}" stroke-width="2" stroke-dasharray="3 2"/><path d="M15 14l7 3-3 1-1 3z" fill="{fg}"/>"#
            ),
            Icon::SelectFree => format!(
                r#"<path d="M3 5C2 14 6 21 12 21S22 14 21 5c-1-3-4-2-4 1 0 5-2 9-5 9S7 11 7 6C7 3 4 2 3 5z" fill="none" stroke="{sel}" stroke-width="2" stroke-dasharray="3 2" stroke-linejoin="round"/><path d="M5 6c1 7 3 12 7 12s6-5 7-12" fill="none" stroke="{found}" stroke-width="1.6" stroke-linecap="round"/>"#
            ),
            Icon::Suppress => format!(
                r#"<rect x="2" y="4" width="20" height="16" fill="none" stroke="{sel}" stroke-width="2" stroke-dasharray="3 2"/><path d="M7 12h10" stroke="{found}" stroke-width="3" stroke-linecap="round"/>"#
            ),
            Icon::Extract => format!(
                r#"<rect x="2" y="6" width="15" height="15" fill="none" stroke="{sel}" stroke-width="2" stroke-dasharray="3 2"/><path d="M9 14L21 3M14 3h7v7" {line}/>"#
            ),
            Icon::LearnNoise => format!(
                r#"<path d="M2 19l1.5-3 1.5 2 1.5-4 1.5 3 1.5-2 1.5 3 1.5-4 1.5 2 1.5-1 1.5 3 1.5-3 1.5 2 1.5-2" {line}/><path d="M2 9h20" stroke="{found}" stroke-width="2" stroke-dasharray="3 2"/>"#
            ),
            Icon::FindSounds => format!(
                r#"<ellipse cx="10" cy="10" rx="3" ry="2" fill="{found}"/><circle cx="10" cy="10" r="7" {line}/><path d="M15 15l7 7" fill="none" stroke="{fg}" stroke-width="3" stroke-linecap="round"/>"#
            ),
            Icon::ZoomFit => format!(
                r#"<path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6" {line}/><path d="M7 12l2-3 2 5 2-4 2 3 2-1" fill="none" stroke="{sel}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>"#
            ),
            Icon::LogFreq => format!(
                r#"<path d="M4 2v18h18" {line}/><path d="M5 19C7 9 12 6 21 5" fill="none" stroke="{sel}" stroke-width="2" stroke-linecap="round"/>"#
            ),
        }
    }

    /// The complete SVG document of the icon.
    pub fn svg(self, p: &Palette) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{ICON_SIZE}" height="{ICON_SIZE}" viewBox="0 0 24 24">{}</svg>"#,
            self.body(p)
        )
    }
}

/// Creates the frame's toolbar from [`ITEMS`]. Clicks reach the frame's
/// `on_menu` handler with the tool's id. Returns the toolbar so radio
/// tools can be kept in step with the program's state.
pub fn create(frame: &Frame) -> Option<ToolBar> {
    let tb = frame.create_tool_bar(Some(ToolBarStyle::Default | ToolBarStyle::Flat), ID_ANY as i32)?;
    let palette = Palette::current();
    for item in ITEMS {
        match *item {
            Item::Tool { id, label, help, icon } => {
                // An icon that fails to render must not cost the command:
                // fall back to an empty bundle (the tooltip still works).
                let bundle = BitmapBundle::from_svg_text(&icon.svg(&palette), Size::new(ICON_SIZE, ICON_SIZE))
                    .unwrap_or_default();
                tb.add_tool_bundle(id, label, &bundle, help);
            }
            Item::Radio { id, label, help, icon } => {
                // wxDragon adds radio tools from a plain bitmap only, so
                // render the bundle at the toolbar's scale (HiDPI).
                let bitmap = BitmapBundle::from_svg_text(&icon.svg(&palette), Size::new(ICON_SIZE, ICON_SIZE))
                    .and_then(|b| b.get_bitmap_for(&tb));
                if let Some(bitmap) = bitmap {
                    tb.add_radio_tool(id, label, &bitmap, help);
                }
            }
            Item::Separator => tb.add_separator(),
        }
    }
    tb.realize();
    Some(tb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The tools of [`ITEMS`] as `(id, label, help, icon)`.
    fn tools() -> Vec<(i32, &'static str, &'static str, Icon)> {
        ITEMS
            .iter()
            .filter_map(|item| match *item {
                Item::Tool { id, label, help, icon } | Item::Radio { id, label, help, icon } => {
                    Some((id, label, help, icon))
                }
                Item::Separator => None,
            })
            .collect()
    }

    #[test]
    fn every_tool_is_unique_and_described() {
        let tools = tools();
        let ids: HashSet<_> = tools.iter().map(|t| t.0).collect();
        assert_eq!(ids.len(), tools.len(), "two tools send the same command");
        let icons: HashSet<_> = tools.iter().map(|t| t.3).collect();
        assert_eq!(icons.len(), tools.len(), "two tools look the same");
        for (_, label, help, _) in &tools {
            assert!(!label.is_empty() && !help.is_empty());
        }
    }

    #[test]
    fn separators_only_between_groups() {
        assert!(!matches!(ITEMS.first(), Some(Item::Separator)));
        assert!(!matches!(ITEMS.last(), Some(Item::Separator)));
        for pair in ITEMS.windows(2) {
            assert!(!(pair[0] == Item::Separator && pair[1] == Item::Separator));
        }
    }

    #[test]
    fn tools_send_menu_commands() {
        // Every tool must be a command the frame's menu handler knows;
        // otherwise its click would silently do nothing.
        let menu_ids = [
            crate::ID_OPEN_AUDIO,
            crate::ID_OPEN_PROJECT,
            crate::ID_SAVE_PROJECT,
            crate::ID_EXPORT,
            crate::ID_PLAY,
            crate::ID_ZOOM_FIT,
            crate::ID_LOG_FREQ,
            crate::ID_ADD_SUPPRESS,
            crate::ID_ADD_EXTRACT,
            crate::ID_LEARN_NOISE,
            crate::ID_FIND_SOUNDS,
            crate::ID_HEAR_ALL,
            crate::ID_HEAR_SOLO,
            crate::ID_HEAR_WITHOUT,
            crate::ID_SELECT_RECT,
            crate::ID_SELECT_FREE,
        ];
        for (id, label, _, _) in tools() {
            assert!(menu_ids.contains(&id), "{label} has no menu command");
        }
    }

    /// Checks that `svg` is a well-formed document as far as nanosvg cares:
    /// every element is closed, attributes are quoted, nothing is left of
    /// the format placeholders.
    fn assert_well_formed(svg: &str) {
        assert!(svg.starts_with("<svg ") && svg.ends_with("</svg>"), "{svg}");
        assert!(
            !svg.contains('{') && !svg.contains('}'),
            "unexpanded placeholder: {svg}"
        );
        assert_eq!(svg.matches('"').count() % 2, 0, "unbalanced quotes: {svg}");
        // Inside the root, every element is self-closing.
        let inner = &svg[svg.find('>').unwrap() + 1..svg.len() - "</svg>".len()];
        assert_eq!(inner.matches('<').count(), inner.matches("/>").count(), "{svg}");
        assert!(svg.contains(r#"viewBox="0 0 24 24""#));
    }

    #[test]
    fn icons_are_well_formed_in_both_palettes() {
        for icon in Icon::ALL {
            for p in [Palette::LIGHT, Palette::DARK] {
                let svg = icon.svg(&p);
                assert_well_formed(&svg);
                assert!(
                    svg.contains(p.fg) || svg.contains(p.selection),
                    "{icon:?} ignores the palette"
                );
            }
        }
    }

    /// The ids of each run of adjacent radio tools: the radio groups
    /// wxWidgets forms.
    fn radio_groups() -> Vec<Vec<i32>> {
        let mut groups: Vec<Vec<i32>> = Vec::new();
        let mut in_group = false;
        for item in ITEMS {
            match *item {
                Item::Radio { id, .. } => {
                    if !in_group {
                        groups.push(Vec::new());
                    }
                    if let Some(group) = groups.last_mut() {
                        group.push(id);
                    }
                    in_group = true;
                }
                _ => in_group = false,
            }
        }
        groups
    }

    #[test]
    fn radio_tools_form_the_menus_radio_groups() {
        // The listening modes and the selection tools are one radio group
        // each: adjacent, in menu order, kept apart.
        let listen: Vec<i32> = crate::LISTEN_MENU.iter().map(|&(_, id)| id).collect();
        let select: Vec<i32> = crate::SELECT_MENU.iter().map(|&(_, id)| id).collect();
        assert_eq!(radio_groups(), vec![listen, select]);
    }

    #[test]
    fn dark_palette_differs_from_light() {
        // Same icon, other appearance: the strokes must change, or the
        // icon would vanish on one of the backgrounds.
        assert_ne!(Icon::Save.svg(&Palette::LIGHT), Icon::Save.svg(&Palette::DARK));
    }
}
