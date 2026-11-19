//! File > Settings: a dialog of pages listed down its left side, each with
//! an icon, like losslessaudiocut's Edit > Settings (keep them alike). It
//! holds what a user sets once and then rarely touches, so the filter
//! pane can keep to the filters.
//!
//! Pages:
//! - **Spectrogram**: the levels drawn black and at full colour.
//! - **Find sounds**: how many spectral patterns the search looks for and
//!   whether patterns that come and go together are suggested as one sound.
//! - **Export**: peak normalization of the exported audio.
//!
//! Changes show at once (the caller's `preview`), so the user sees what a
//! level does to the spectrogram; Cancel goes back to what was there
//! before. The rules (ranges, the floor staying below the ceiling) are
//! [`Options`]'; the dialog only shows them.

use std::cell::Cell;
use std::rc::Rc;

use noiseextractor_core::normalize::PEAK_DBFS_RANGE;
use wxdragon::prelude::*;

use crate::options::{tenths, Options, DB_RANGE, PATTERN_RANGE};
use crate::toolbar::{Icon, Palette};

/// Size of the page icons in the list, in pixels.
const ICON: i32 = 32;
/// Width the explanations wrap at, in pixels.
const TEXT_WIDTH: i32 = 380;
/// Space around and between the controls, in pixels.
const GAP: i32 = 8;
/// Smallest height of the page list, in pixels: room for all page tabs.
const BOOK_MIN_HEIGHT: i32 = 300;

/// The pages, in order, with their titles and icons (shared with the
/// toolbar's commands of the same topic).
pub const PAGES: [(&str, Icon); 3] = [
    ("Spectrogram", Icon::LogFreq),
    ("Find sounds", Icon::FindSounds),
    ("Export", Icon::Export),
];

/// Shared state of an open dialog: the options as currently chosen and
/// the callback that shows them in the main window.
#[derive(Clone)]
struct Chosen {
    /// The options as the controls show them.
    options: Rc<Cell<Options>>,
    /// Applies options to the main window right away.
    preview: Rc<dyn Fn(Options)>,
}

impl Chosen {
    /// Applies `change` to the chosen options, settles them by their rules,
    /// previews them and returns the result (so the caller can show where a
    /// rule moved a value).
    fn update(&self, change: impl FnOnce(Options) -> Options) -> Options {
        let o = change(self.options.get()).normalized();
        self.options.set(o);
        (self.preview)(o);
        o
    }
}

/// Shows the dialog over `parent`, starting from `current`. Every change
/// is passed to `preview` right away; returns the options to keep on OK,
/// `None` on Cancel (the caller then previews `current` again).
pub fn show(parent: &dyn WxWidget, current: Options, preview: Rc<dyn Fn(Options)>) -> Option<Options> {
    let dialog = Dialog::builder(parent, "Settings")
        .with_style(DialogStyle::DefaultDialogStyle | DialogStyle::ResizeBorder)
        .build();
    let chosen = Chosen {
        options: Rc::new(Cell::new(current)),
        preview,
    };
    let book = Notebook::builder(&dialog).with_style(NotebookStyle::Left).build();
    // GTK draws left-hand tabs rotated, each as tall as its label; without
    // this the dialog is only as tall as the tallest page and hides all
    // but one tab behind scroll arrows.
    book.set_min_size(Size::new(560, BOOK_MIN_HEIGHT));
    let images = ImageList::new(ICON, ICON, true, PAGES.len() as i32);
    let palette = Palette::current();
    for (_, icon) in PAGES {
        let size = Size::new(ICON, ICON);
        // A missing icon only leaves the tab without a picture.
        if let Some(bitmap) = BitmapBundle::from_svg_text(&icon.svg(&palette), size).and_then(|b| b.get_bitmap(size)) {
            images.add_bitmap(&bitmap);
        }
    }
    book.set_image_list(images);
    let pages = [
        spectrogram_page(&book, &chosen),
        find_sounds_page(&book, &chosen),
        export_page(&book, &chosen),
    ];
    for (i, (page, (title, _))) in pages.iter().zip(PAGES).enumerate() {
        book.add_page(page, title, i == 0, Some(i as i32));
    }

    let buttons = StdDialogButtonSizerBuilder::new().build();
    // wxDialog closes itself for these two ids, with them as the result.
    buttons.add_button(&Button::builder(&dialog).with_id(ID_OK).with_label("OK").build());
    buttons.add_button(&Button::builder(&dialog).with_id(ID_CANCEL).with_label("Cancel").build());
    buttons.realize();
    let sizer = BoxSizer::builder(Orientation::Vertical).build();
    sizer.add(&book, 1, SizerFlag::Expand | SizerFlag::All, GAP);
    sizer.add_sizer(&buttons, 0, SizerFlag::Expand | SizerFlag::All, GAP);
    dialog.set_sizer_and_fit(sizer, true);

    let result = (dialog.show_modal() == ID_OK).then(|| chosen.options.get());
    dialog.destroy();
    result
}

/// An explanation under a control, wrapped to [`TEXT_WIDTH`].
fn note(page: &Panel, text: &str) -> StaticText {
    let t = StaticText::builder(page).with_label(text).build();
    t.wrap(TEXT_WIDTH);
    t
}

/// A whole-number spin control over `range`, showing `value`.
fn spin(page: &Panel, range: (i32, i32), value: i32) -> SpinCtrl {
    let s = SpinCtrl::builder(page).with_range(range.0, range.1).build();
    s.set_value(value);
    s
}

/// Adds `label` and `control` side by side as one row of `sizer`.
fn row(page: &Panel, sizer: &BoxSizer, label: &str, control: &impl WxWidget) {
    let r = BoxSizer::builder(Orientation::Horizontal).build();
    r.add(
        &StaticText::builder(page).with_label(label).build(),
        1,
        SizerFlag::AlignCenterVertical | SizerFlag::Right,
        GAP,
    );
    r.add(control, 0, SizerFlag::AlignCenterVertical, 0);
    sizer.add_sizer(&r, 0, SizerFlag::Expand | SizerFlag::All, GAP);
}

/// The "Spectrogram" page: the level drawn black and the one drawn at full
/// colour. Moving either redraws the spectrogram at once.
fn spectrogram_page(book: &Notebook, chosen: &Chosen) -> Panel {
    let page = Panel::builder(book).build();
    let o = chosen.options.get();
    let (lo, hi) = (*DB_RANGE.start(), *DB_RANGE.end());
    let floor = spin(&page, (lo, hi), o.db_floor);
    let ceiling = spin(&page, (lo, hi), o.db_ceiling);
    // Shows where the rules put both values (moving one may push the other).
    let show = move |o: Options| {
        floor.set_value(o.db_floor);
        ceiling.set_value(o.db_ceiling);
    };
    {
        let chosen = chosen.clone();
        floor.on_value_changed(move |e| show(chosen.update(|o| o.with_floor(e.get_value()))));
    }
    {
        let chosen = chosen.clone();
        ceiling.on_value_changed(move |e| show(chosen.update(|o| o.with_ceiling(e.get_value()))));
    }
    let defaults = Button::builder(&page).with_label("Default levels").build();
    {
        let chosen = chosen.clone();
        defaults.on_click(move |_| {
            let d = Options::default();
            show(chosen.update(|o| o.with_floor(d.db_floor).with_ceiling(d.db_ceiling)));
        });
    }

    let sizer = BoxSizer::builder(Orientation::Vertical).build();
    row(&page, &sizer, "Black below (dB)", &floor);
    row(&page, &sizer, "Full colour from (dB)", &ceiling);
    sizer.add(
        &note(
            &page,
            "Lower the first value to see quieter detail (and more of the noise \
             floor); lower the second to make quiet recordings brighter. Only the \
             picture changes, not the sound.",
        ),
        0,
        SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom,
        GAP,
    );
    sizer.add(&defaults, 0, SizerFlag::All, GAP);
    page.set_sizer_and_fit(sizer, true);
    page
}

/// The "Find sounds" page: the settings of Filter > Find sounds.
fn find_sounds_page(book: &Notebook, chosen: &Chosen) -> Panel {
    let page = Panel::builder(book).build();
    let o = chosen.options.get();
    let patterns = spin(
        &page,
        (*PATTERN_RANGE.start() as i32, *PATTERN_RANGE.end() as i32),
        o.patterns as i32,
    );
    {
        let chosen = chosen.clone();
        patterns.on_value_changed(move |e| {
            chosen.update(|o| Options {
                patterns: u32::try_from(e.get_value()).unwrap_or(1),
                ..o
            });
        });
    }
    let group = CheckBox::builder(&page)
        .with_label("Group patterns by timing and direction")
        .build();
    group.set_value(o.group);
    {
        let chosen = chosen.clone();
        group.on_toggled(move |e| {
            chosen.update(|o| Options {
                group: e.is_checked(),
                ..o
            });
        });
    }

    let sizer = BoxSizer::builder(Orientation::Vertical).build();
    row(&page, &sizer, "Patterns to look for", &patterns);
    sizer.add(
        &note(
            &page,
            "Each pattern is one spectral shape that keeps coming back. More \
             patterns tell more sounds apart but take longer and may split one \
             sound into several.",
        ),
        0,
        SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom,
        GAP,
    );
    sizer.add(&group, 0, SizerFlag::All, GAP);
    sizer.add(
        &note(
            &page,
            "Suggests patterns that come and go together (and, in stereo, from \
             the same direction) as one sound, e.g. a call and its overtones.",
        ),
        0,
        SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom,
        GAP,
    );
    page.set_sizer_and_fit(sizer, true);
    page
}

/// The "Export" page: peak normalization of File > Export.
fn export_page(book: &Notebook, chosen: &Chosen) -> Panel {
    let page = Panel::builder(book).build();
    let o = chosen.options.get();
    let normalize = CheckBox::builder(&page)
        .with_label("Normalize the exported audio")
        .build();
    normalize.set_value(o.normalize);
    let peak = SpinCtrlDouble::builder(&page)
        .with_range(f64::from(PEAK_DBFS_RANGE.0), f64::from(PEAK_DBFS_RANGE.1))
        .build();
    peak.set_increment(0.5);
    peak.set_digits(1);
    peak.set_value(f64::from(o.peak_dbfs()));
    peak.enable(o.normalize);
    {
        let chosen = chosen.clone();
        normalize.on_toggled(move |e| {
            let o = chosen.update(|o| Options {
                normalize: e.is_checked(),
                ..o
            });
            peak.enable(o.normalize);
        });
    }
    {
        let chosen = chosen.clone();
        peak.on_value_changed(move |_| {
            chosen.update(|o| Options {
                peak_dbfs_tenths: tenths(peak.get_value() as f32),
                ..o
            });
        });
    }

    let sizer = BoxSizer::builder(Orientation::Vertical).build();
    sizer.add(&normalize, 0, SizerFlag::All, GAP);
    row(&page, &sizer, "Loudest sample at (dBFS)", &peak);
    sizer.add(
        &note(
            &page,
            "Raises (or lowers) the whole export by one gain, so that its \
             loudest sample reaches this level. An extracted sound is often \
             much quieter than the recording; this makes it usable without \
             changing how loud its parts are to each other. -1 dBFS leaves \
             room for converting to MP3 or Ogg later.",
        ),
        0,
        SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom,
        GAP,
    );
    page.set_sizer_and_fit(sizer, true);
    page
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_have_distinct_titles_and_icons() {
        for (i, (title, icon)) in PAGES.iter().enumerate() {
            for (other, other_icon) in &PAGES[i + 1..] {
                assert_ne!(title, other);
                assert_ne!(icon, other_icon);
            }
        }
    }
}
