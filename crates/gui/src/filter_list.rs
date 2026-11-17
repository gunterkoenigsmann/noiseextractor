//! The filter list: one row per filter, with a box that switches the
//! filter on or off, and several rows can be picked at once (Ctrl+click,
//! Shift+click) to hear them alone or removed together.
//!
//! wxDragon's `CheckListBox` cannot report a multiple selection, so this is
//! a one-column report-style `ListCtrl` whose item icons are drawn check
//! boxes: a click on the icon (or Space) switches the filter, a click on
//! the text picks it. What the list shows and which rows a pick means are
//! decided by the pure functions here, so they are testable without a
//! display.

use wxdragon::prelude::*;
use wxdragon::widgets::imagelist::ImageList;
use wxdragon::widgets::list_ctrl::{image_list_type, ListItemState, ListNextItemFlag};

/// `wxLIST_HITTEST_ONITEMICON`: the hit-test result flag for a click on the
/// item's icon, which is our check box (wx/listbase.h; same value in the
/// native MSW and the generic GTK/macOS implementation).
const HITTEST_ONITEMICON: i32 = 0x0020;

/// Side of the check box icons in pixels.
const BOX: i32 = 16;

/// Image list index of the empty box.
const IMAGE_OFF: i32 = 0;
/// Image list index of the ticked box.
const IMAGE_ON: i32 = 1;

/// The filter list widget.
#[derive(Clone, Copy)]
pub struct FilterList {
    /// The underlying control.
    ctrl: ListCtrl,
}

impl FilterList {
    /// Creates the list in `parent`.
    pub fn new(parent: &dyn WxWidget, size: Size) -> Self {
        let ctrl = ListCtrl::builder(parent)
            .with_style(ListCtrlStyle::Report | ListCtrlStyle::NoHeader)
            .with_size(size)
            .build();
        ctrl.insert_column(0, "Filter", ListColumnFormat::Left, size.width.max(120) - 8);
        let images = ImageList::new(BOX, BOX, true, 2);
        for on in [false, true] {
            if let Some(b) = check_box_bitmap(on) {
                images.add_bitmap(&b);
            }
        }
        ctrl.set_image_list(images, image_list_type::SMALL);
        Self { ctrl }
    }

    /// The wrapped control, for sizers and generic window calls.
    pub fn ctrl(&self) -> ListCtrl {
        self.ctrl
    }

    /// Replaces the rows: `(label, enabled)` per filter. `picked` rows are
    /// selected, `focused` gets the keyboard focus and is scrolled to.
    pub fn set_rows(&self, rows: &[(String, bool)], picked: &[usize], focused: Option<usize>) {
        self.ctrl.delete_all_items();
        for (i, (label, on)) in rows.iter().enumerate() {
            self.ctrl
                .insert_item(i as i64, label, Some(if *on { IMAGE_ON } else { IMAGE_OFF }));
        }
        for &i in picked.iter().filter(|&&i| i < rows.len()) {
            self.ctrl
                .set_item_state(i as i64, ListItemState::Selected, ListItemState::Selected);
        }
        if let Some(i) = focused.filter(|&i| i < rows.len()) {
            self.ctrl
                .set_item_state(i as i64, ListItemState::Focused, ListItemState::Focused);
            self.ctrl.ensure_visible(i as i64);
        }
    }

    /// Shows filter `index` as on or off.
    pub fn set_enabled(&self, index: usize, on: bool) {
        self.ctrl
            .set_item_image(index as i64, if on { IMAGE_ON } else { IMAGE_OFF });
    }

    /// The selected rows, in list order.
    pub fn picked(&self) -> Vec<usize> {
        let mut rows = Vec::new();
        let mut i = -1i64;
        loop {
            let next = self
                .ctrl
                .get_next_item(i, ListNextItemFlag::All, ListItemState::Selected);
            if next < 0 {
                return rows;
            }
            rows.push(next as usize);
            i = next as i64;
        }
    }

    /// Calls `f(picked rows, row just clicked)` whenever the selection
    /// changes. wxWidgets sends one event per row that changes, so a
    /// Shift+click arrives several times; every call sees the whole
    /// selection as it is at that moment.
    pub fn on_pick(&self, f: impl Fn(Vec<usize>, Option<usize>) + Clone + 'static) {
        let list = *self;
        let g = f.clone();
        self.ctrl.on_item_selected(move |e| {
            let row = usize::try_from(e.get_item_index()).ok();
            g(list.picked(), row);
        });
        self.ctrl.on_item_deselected(move |_| f(list.picked(), None));
    }

    /// Calls `f(rows)` when the user asks to switch the filters in `rows`
    /// on or off: a click on a row's box (that row) or Space (all picked
    /// rows).
    pub fn on_toggle(&self, f: impl Fn(Vec<usize>) + Clone + 'static) {
        let list = *self;
        let g = f.clone();
        MouseEvents(self.ctrl).on_mouse_left_down(move |e| {
            // The control picks the clicked row as usual, box or not (as
            // a check list box does).
            e.skip(true);
            if let WindowEventData::MouseButton(m) = &e {
                if let Some(pos) = m.get_position() {
                    let (row, flags, _) = list.ctrl.hit_test(pos);
                    if row >= 0 && flags & HITTEST_ONITEMICON != 0 {
                        g(vec![row as usize]);
                    }
                }
            }
        });
        self.ctrl.on_key_down(move |e| {
            if e.get_key_code() == Some(WXK_SPACE) {
                f(list.picked());
            }
        });
    }
}

/// The list control as a receiver of plain window (mouse) events, which
/// wxDragon's `ListCtrl` does not offer itself. On GTK and macOS the rows
/// live in a child window, which hands its mouse events to the control
/// first (wxGenericListCtrl), so binding them here works everywhere.
struct MouseEvents(ListCtrl);

impl WxEvtHandler for MouseEvents {
    unsafe fn get_event_handler_ptr(&self) -> *mut wxdragon::ffi::wxd_EvtHandler_t {
        // SAFETY: forwards to the control's own implementation; the
        // pointer is null once the control is destroyed, which binding
        // checks.
        unsafe { self.0.get_event_handler_ptr() }
    }
}

impl WindowEvents for MouseEvents {}

/// The rows a pick refers to: the selected rows if they include the
/// focused filter, else just the focused filter. The list's selection can
/// be out of date when the program moved the focus (a new, moved or
/// deleted filter); the focused filter is always right.
pub fn effective_picks(selected: &[usize], focused: Option<usize>) -> Vec<usize> {
    match focused {
        Some(f) if selected.contains(&f) => selected.to_vec(),
        Some(f) => vec![f],
        None => Vec::new(),
    }
}

/// The new on/off state when the user toggles several filters at once:
/// all on, unless all of them are on already (as file managers do).
pub fn toggled_state(states: &[bool]) -> bool {
    !states.iter().all(|&on| on)
}

/// SVG of a check box, ticked or empty, in neutral colours that work on
/// light and dark list backgrounds.
fn check_box_svg(on: bool) -> String {
    let tick = if on {
        r##"<path d="M4 8.5l3 3 5-6" fill="none" stroke="#1e8c32" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>"##
    } else {
        ""
    };
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{BOX}" height="{BOX}" viewBox="0 0 16 16"><rect x="1.5" y="1.5" width="13" height="13" rx="2" fill="#ffffff" stroke="#808080" stroke-width="1.2"/>{tick}</svg>"##
    )
}

/// The check box icon as a bitmap.
fn check_box_bitmap(on: bool) -> Option<Bitmap> {
    BitmapBundle::from_svg_text(&check_box_svg(on), Size::new(BOX, BOX))?.get_bitmap(Size::new(BOX, BOX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_fall_back_to_the_focused_filter() {
        assert_eq!(effective_picks(&[1, 3], Some(3)), vec![1, 3]);
        // The program focused another filter: the old selection is stale.
        assert_eq!(effective_picks(&[1, 3], Some(2)), vec![2]);
        assert_eq!(effective_picks(&[], Some(0)), vec![0]);
        assert!(effective_picks(&[1], None).is_empty());
    }

    #[test]
    fn toggling_several_switches_all_on_unless_all_are_on() {
        assert!(toggled_state(&[false, true]));
        assert!(toggled_state(&[false]));
        assert!(!toggled_state(&[true, true]));
    }

    #[test]
    fn check_boxes_differ_and_are_well_formed() {
        let (off, on) = (check_box_svg(false), check_box_svg(true));
        assert_ne!(off, on);
        for svg in [off, on] {
            assert!(svg.starts_with("<svg ") && svg.ends_with("</svg>"));
            assert!(!svg.contains('{'), "unexpanded placeholder: {svg}");
        }
    }
}
