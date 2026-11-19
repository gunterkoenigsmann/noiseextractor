//! noiseextractor GUI: an interactive spectrogram in which noises can be
//! marked, inspected and suppressed or extracted with a list of filters.
//!
//! Layout: dockable panes (wxWidgets AUI, see `panes.rs`) around the
//! spectrogram: the filter list ("building kit") and the detection
//! settings of the selected sound on the right, a waveform overview on
//! top, a log at the bottom. All heavy lifting happens in
//! `noiseextractor-core` on background threads; this crate only wires the
//! widgets to it. A 40 ms timer polls the background work (file loading,
//! selection analysis, chunk processing, playback position) and repaints
//! when something changed.
//!
//! Mouse and keys on the spectrogram:
//! * drag: mark a rectangle (time × frequency); the analysis immediately
//!   shows what it thinks belongs to it (red / orange),
//! * click: move the play head,
//! * right-button drag: scrub (hear short pieces of audio under the
//!   pointer, see `scrub.rs`),
//! * wheel: zoom time, Shift+wheel: scroll, Ctrl+wheel: zoom frequency,
//! * Space: play / pause (the selection, if there is one), Esc: clear the
//!   selection.
//!
//! In the overview, dragging the frame (or clicking beside it) scrolls the
//! spectrogram; dragging beside the frame selects a time span and zooms the
//! spectrogram to it. An orange envelope there shows how loud what the
//! selection being edited (else the filter picked in the list) finds is at
//! each time, so repetitions of a marked sound can be spotted in the whole
//! recording.
//!
//! Picking a filter in the list draws its overlays even when "Show in
//! spectrogram" is off: what an event filter targets in green / red /
//! orange, what a noise reduction takes away in red.
//!
//! View > Hear ... (Ctrl+1/2/3) chooses what is played and drawn in blue:
//! everything, only the picked sound (the selection being edited, else the
//! filters picked in the list; Ctrl/Shift+click picks several), or
//! everything but it. The mode stays until
//! changed and follows the pick; see `noiseextractor_core::listen`. Export
//! always writes the project as set up, whatever is being heard.
//!
//! "Find sounds" looks for the recurring sounds the recording (or the
//! selected stretch) is made of (see `noiseextractor_core::discover`) and
//! adds them to the list as suggestions: switched off and hidden until
//! picked. They can be kept, renamed, merged, split or discarded; their
//! parts are the spectral patterns found.

// A release build is a GUI program on Windows: started from Explorer it
// gets no console window. Debug builds keep the console for their output.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// Every item, public or private, carries a doc comment (project rule).
#![warn(missing_docs, clippy::missing_docs_in_private_items)]

mod detection;
mod filter_list;
mod log;
mod overview;
mod panes;
mod player;
mod recent;
mod scrub;
mod toolbar;
mod view;

use detection::{Group, Param};
use filter_list::FilterList;
use noiseextractor_core::analysis::{analyze_selection, AnalysisParams, RegionAnalysis, Selection};
use noiseextractor_core::audio::WavStreamWriter;
use noiseextractor_core::discover::{discover, DiscoverParams, Discovery};
use noiseextractor_core::engine::process_all;
use noiseextractor_core::filter::{EventFilter, Filter, FilterAction, FilterKind, Project};
use noiseextractor_core::noise::{mean_power, NoiseEstimate};
use noiseextractor_core::render::{render_rgb, ORANGE};
use noiseextractor_core::similar::SimilarParams;
use noiseextractor_core::stft::{Stft, StftConfig};
use noiseextractor_core::{AudioData, Engine, EngineConfig, Focus, Listen, Pick};
use overview::{found_columns, Overview, Peaks};
use panes::{Contents, Pane, Panes};
use player::Player;
use recent::{Command as RecentCommand, Kind as RecentKind, RecentFiles};
use scrub::Scrubber;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};
use view::{format_time, Viewport};
use wxdragon::prelude::*;

// Menu command ids.
/// File > Open audio.
const ID_OPEN_AUDIO: i32 = ID_HIGHEST + 1;
/// File > Open project.
const ID_OPEN_PROJECT: i32 = ID_HIGHEST + 2;
/// File > Save project.
const ID_SAVE_PROJECT: i32 = ID_HIGHEST + 3;
/// File > Export.
const ID_EXPORT: i32 = ID_HIGHEST + 4;
/// View > Play / pause.
const ID_PLAY: i32 = ID_HIGHEST + 5;
/// View > Show whole recording.
const ID_ZOOM_FIT: i32 = ID_HIGHEST + 6;
/// View > Logarithmic frequency axis.
const ID_LOG_FREQ: i32 = ID_HIGHEST + 7;
/// File > Quit.
const ID_QUIT: i32 = ID_HIGHEST + 8;
/// View > Hear everything ([`Listen::All`]).
const ID_HEAR_ALL: i32 = ID_HIGHEST + 9;
/// View > Hear the picked sound only ([`Listen::Solo`]).
const ID_HEAR_SOLO: i32 = ID_HIGHEST + 10;
/// View > Hear everything but the picked sound ([`Listen::Without`]).
const ID_HEAR_WITHOUT: i32 = ID_HIGHEST + 11;

/// Filter > Suppress selection (also a button in the filter pane).
const ID_ADD_SUPPRESS: i32 = ID_HIGHEST + 20;
/// Filter > Extract selection (also a button in the filter pane).
const ID_ADD_EXTRACT: i32 = ID_HIGHEST + 21;
/// Filter > Learn noise from selection (also a button in the filter pane).
const ID_LEARN_NOISE: i32 = ID_HIGHEST + 22;
/// Filter > Find sounds (also a button in the filter pane).
const ID_FIND_SOUNDS: i32 = ID_HIGHEST + 23;

/// The listening modes with their menu ids, in menu order.
const LISTEN_MENU: [(Listen, i32); 3] = [
    (Listen::All, ID_HEAR_ALL),
    (Listen::Solo, ID_HEAR_SOLO),
    (Listen::Without, ID_HEAR_WITHOUT),
];

/// Supported input formats for the open dialog (from
/// `noiseextractor_core::audio::FILE_TYPES`, which the packages' file
/// associations are checked against).
fn audio_wildcard() -> String {
    format!(
        "Audio files|{}|All files|*",
        noiseextractor_core::audio::file_dialog_patterns()
    )
}
/// Project files.
const PROJECT_WILDCARD: &str = "noiseextractor projects (*.nxp.json)|*.nxp.json;*.json|All files|*";

/// A loaded recording with its processing engine and output stream.
struct Doc {
    /// The recording.
    audio: Arc<AudioData>,
    /// Background processing of the recording.
    engine: Arc<Engine>,
    /// `None` when no output device is available.
    player: Option<Player>,
    /// File name for the title bar.
    name: String,
    /// Block peaks of the mono mix, for the overview.
    peaks: Peaks,
}

/// Messages from background threads to the GUI thread.
enum Background {
    /// A file finished loading (or failed), with the overview's peaks.
    Loaded(PathBuf, anyhow::Result<(AudioData, Peaks)>),
    /// An analysis finished. It belongs to every event filter (and the
    /// preview) that still has this selection and these settings.
    Analysed(Selection, AnalysisParams, Arc<RegionAnalysis>),
    /// Export progress in percent, or the final result.
    ExportProgress(usize),
    /// Export finished.
    Exported(anyhow::Result<PathBuf>),
    /// The search for the recording's sounds finished.
    Discovered(anyhow::Result<Discovery>),
}

/// All mutable application state, owned by the GUI thread.
struct State {
    /// The loaded recording, if any.
    doc: Option<Doc>,
    /// The filter list being edited.
    project: Project,
    /// Where the project was loaded from / saved to.
    project_path: Option<PathBuf>,
    /// Visible window onto the recording.
    viewport: Option<Viewport>,
    /// The current marked rectangle.
    selection: Option<Selection>,
    /// Mouse drag in progress: start and current pixel positions.
    drag: Option<((f32, f32), (f32, f32))>,
    /// When the selected object is to be analysed again after its
    /// detection settings changed (debounced while a slider moves).
    reanalyse_at: Option<Instant>,
    /// Detection settings last chosen with the sliders; new selections
    /// start with them.
    last_params: (AnalysisParams, SimilarParams),
    /// The scrub in progress while the right button is held.
    scrub: Option<Scrub>,
    /// The overview strip's view model (exists with a recording).
    overview: Option<Overview>,
    /// Sender handed to background threads.
    tx: Sender<Background>,
    /// Receiver polled by the timer.
    rx: Receiver<Background>,
    /// Index of the filter shown in the detail controls.
    current_filter: Option<usize>,
    /// Rows picked in the filter list (several with Ctrl/Shift+click);
    /// only meaningful while they include [`State::current_filter`], see
    /// [`filter_list::effective_picks`].
    picked: Vec<usize>,
    /// Engine revision at the last repaint.
    last_revision: u64,
    /// Play head (samples) when not playing.
    playhead: usize,
    /// Number used for default filter names.
    filter_counter: usize,
    /// Status shown while something runs in the background.
    busy: Option<String>,
    /// What is heard (and drawn in blue): everything, only the picked
    /// sound, or everything but it (see [`listen_picks`]).
    listen: Listen,
    /// File > Recent recordings, remembered between runs.
    recent_audio: RecentFiles,
    /// File > Recent projects, remembered between runs.
    recent_projects: RecentFiles,
}

impl State {
    /// Shows filter `index` in the details and makes it the only pick
    /// (used when the program, not a click in the list, moves the focus).
    fn focus_filter(&mut self, index: Option<usize>) {
        self.current_filter = index;
        self.picked = index.into_iter().collect();
    }

    /// The filters the listening modes refer to (see [`listen_picks`]).
    fn listen_picks(&self) -> Vec<Pick> {
        listen_picks(
            &filter_list::effective_picks(&self.picked, self.current_filter),
            self.project.preview.is_some(),
        )
    }

    /// The recent-files list of `kind`.
    fn recent_mut(&mut self, kind: RecentKind) -> &mut RecentFiles {
        match kind {
            RecentKind::Audio => &mut self.recent_audio,
            RecentKind::Project => &mut self.recent_projects,
        }
    }

    /// Puts `path` at the top of the list of `kind`, stores the list and
    /// updates its menu.
    fn remember(&mut self, ui: Ui, kind: RecentKind, path: &Path) {
        let list = self.recent_mut(kind);
        list.add(path.to_path_buf());
        show_recent(ui, kind, list);
    }
}

/// A scrub: dragging with the right button plays short pieces of audio at
/// the pointer (see [`scrub`]).
struct Scrub {
    /// Decides when to restart where.
    scrubber: Scrubber,
    /// Whether playback ran before; if not, it stops when the scrub ends.
    was_playing: bool,
}

/// Debounce of re-analysis while a detection slider moves.
const REANALYSE_DELAY: Duration = Duration::from_millis(200);

/// Handles to the widgets. wxDragon widgets are cheap `Copy` handles.
#[derive(Clone, Copy)]
struct Ui {
    /// Main window.
    frame: Frame,
    /// The toolbar (`None` if it could not be created).
    toolbar: Option<ToolBar>,
    /// The dockable panes.
    panes: Panes,
    /// Spectrogram drawing area.
    canvas: Panel,
    /// Waveform overview drawing area.
    overview: Panel,
    /// Which object the detection sliders edit.
    detection_title: StaticText,
    /// One slider and value label per [`Param::ALL`] entry.
    detection: [(Slider, StaticText); Param::ALL.len()],
    /// The filter list; ticked = enabled, several rows can be picked.
    filters: FilterList,
    /// Name of the current filter.
    name: TextCtrl,
    /// Overlay visibility of the current filter.
    visible: CheckBox,
    /// Suppress / extract of the current filter.
    action: Choice,
    /// Parts of the current filter; unchecked = temporarily disabled.
    parts: CheckListBox,
    /// Maximum attenuation of a noise-reduction filter.
    reduction: Slider,
    /// Shows the slider value.
    reduction_label: StaticText,
    /// Adds the selection as a suppress filter.
    add_suppress: Button,
    /// Adds the selection as an extract filter.
    add_extract: Button,
    /// Removes the current filter.
    delete: Button,
    /// Moves the current filter up.
    up: Button,
    /// Moves the current filter down.
    down: Button,
    /// Learns a noise profile from the selection.
    learn: Button,
    /// Switches back to the adaptive noise estimate.
    adaptive: Button,
    /// Starts the search for the recording's sounds.
    find_sounds: Button,
    /// How many spectral patterns to look for.
    sounds: Slider,
    /// Shows the slider value.
    sounds_label: StaticText,
    /// Whether patterns that belong together are suggested as one sound.
    group: CheckBox,
    /// Keeps the current suggestion (the next search will not replace it).
    keep: Button,
    /// Merges another found sound into the current one.
    merge: Button,
    /// Splits the current found sound into its patterns.
    split: Button,
    /// Removes all suggestions that were not kept.
    discard: Button,
}

/// The application state as shared by all event handlers (GUI thread only).
type Shared = Rc<RefCell<State>>;

thread_local! {
    /// Set when the canvas must be repainted; the timer then refreshes it
    /// once no state borrow is active (see the paint handler).
    static REPAINT_PENDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Requests a repaint of the spectrogram at the next timer tick. Safe to
/// call while the state is borrowed.
fn request_repaint() {
    REPAINT_PENDING.with(|p| p.set(true));
}

/// Program entry: builds the window and runs the event loop. An optional
/// command-line argument names an audio file to open.
fn main() {
    let _ = wxdragon::main(|_| {
        let (tx, rx) = channel();
        let state: Shared = Rc::new(RefCell::new(State {
            doc: None,
            project: Project::default(),
            project_path: None,
            viewport: None,
            selection: None,
            drag: None,
            reanalyse_at: None,
            last_params: Default::default(),
            scrub: None,
            overview: None,
            tx,
            rx,
            current_filter: Some(0),
            picked: vec![0],
            last_revision: 0,
            playhead: 0,
            filter_counter: 1,
            busy: None,
            listen: Listen::All,
            recent_audio: recent::load(RecentKind::Audio),
            recent_projects: recent::load(RecentKind::Project),
        }));
        let ui = build_ui();
        {
            let s = state.borrow();
            show_recent(ui, RecentKind::Audio, &s.recent_audio);
            show_recent(ui, RecentKind::Project, &s.recent_projects);
        }
        bind_events(&state, ui);
        refresh_filter_list(&state.borrow(), ui);
        if let Some(path) = std::env::args().nth(1) {
            start_loading(&mut state.borrow_mut(), PathBuf::from(path));
        }
        ui.frame.show(true);
        ui.frame.centre();
    });
}

/// Creates the frame, menus, and the panes: spectrogram canvas, filter
/// list, detection sliders, overview and log.
fn build_ui() -> Ui {
    let frame = Frame::builder()
        .with_title("noiseextractor")
        .with_size(panes::stored_size())
        .build();

    let file_menu = Menu::builder()
        .append_item(ID_OPEN_AUDIO, "&Open audio...\tCtrl+O", "Load a recording")
        .append_item(ID_OPEN_PROJECT, "Open &project...", "Load a filter project")
        .append_item(ID_SAVE_PROJECT, "&Save project...\tCtrl+S", "Save the filter list")
        .append_separator()
        .append_item(
            ID_EXPORT,
            "&Export processed audio...\tCtrl+E",
            "Write the filtered recording as WAV",
        )
        .append_separator()
        .append_item(ID_QUIT, "&Quit\tCtrl+Q", "Leave noiseextractor")
        .build();
    // The builder cannot add submenus: each list goes right below the
    // command that opens that kind of file (positions count the items
    // inserted before).
    file_menu.insert_submenu(
        1,
        recent::submenu(RecentKind::Audio),
        RecentKind::Audio.menu_label(),
        RecentKind::Audio.menu_help(),
    );
    file_menu.insert_submenu(
        3,
        recent::submenu(RecentKind::Project),
        RecentKind::Project.menu_label(),
        RecentKind::Project.menu_help(),
    );
    let mut view_menu = Menu::builder()
        .append_item(
            ID_PLAY,
            "&Play / pause\tCtrl+P",
            "Play the selection or from the play head",
        )
        .append_radio_item(
            ID_HEAR_ALL,
            "Hear &everything\tCtrl+1",
            "Play the output of all enabled filters",
        )
        .append_radio_item(
            ID_HEAR_SOLO,
            "Hear the picked sound &only\tCtrl+2",
            "Play only the sound of the selection or of the filter picked in the list",
        )
        .append_radio_item(
            ID_HEAR_WITHOUT,
            "Hear everything &but the picked sound\tCtrl+3",
            "Play the output with the sound of the selection or of the picked filter removed",
        )
        .append_separator()
        .append_item(ID_ZOOM_FIT, "Show &whole recording\tCtrl+0", "Zoom out completely")
        .append_item(
            ID_LOG_FREQ,
            "&Logarithmic frequency axis\tCtrl+L",
            "Toggle linear / logarithmic frequency axis",
        )
        .append_separator();
    // The filter pane's most used buttons, as menu commands so the toolbar
    // and keyboard can reach them too.
    let filter_menu = Menu::builder()
        .append_item(
            ID_ADD_SUPPRESS,
            "&Suppress selection",
            "Add a filter that suppresses the selected sound",
        )
        .append_item(
            ID_ADD_EXTRACT,
            "E&xtract selection",
            "Add a filter that keeps only the selected sound",
        )
        .append_separator()
        .append_item(
            ID_LEARN_NOISE,
            "&Learn noise from selection",
            "Learn the background noise from the selection",
        )
        .append_item(
            ID_FIND_SOUNDS,
            "&Find sounds\tCtrl+F",
            "Find the recording's sounds, in the selection if there is one",
        )
        .build();
    for pane in Pane::ALL {
        view_menu = view_menu.append_check_item(pane.menu_id(), pane.menu_label(), pane.menu_help());
    }
    frame.set_menu_bar(
        MenuBar::builder()
            .append(file_menu, "&File")
            .append(view_menu.build(), "&View")
            .append(filter_menu, "Fi&lter")
            .build(),
    );
    let toolbar = toolbar::create(&frame);
    StatusBar::builder(&frame)
        .with_fields_count(3)
        .with_status_widths(vec![-1, 260, 220])
        .add_initial_text(0, "Open an audio file to start (Ctrl+O)")
        .build();

    let canvas = Panel::builder(&frame).build();
    canvas.set_background_style(BackgroundStyle::Paint);
    let overview = Panel::builder(&frame).build();
    overview.set_background_style(BackgroundStyle::Paint);
    let log = log::create(&frame);

    // Scrolled, so the docked pane works at any height.
    let side = ScrolledWindow::builder(&frame).build();
    side.set_scroll_rate(0, 10);
    let filters = FilterList::new(&side, Size::new(280, 140));
    filters.ctrl().set_min_size(Size::new(200, 90));
    let add_suppress = Button::builder(&side).with_label("Suppress selection").build();
    let add_extract = Button::builder(&side).with_label("Extract selection").build();
    let delete = Button::builder(&side).with_label("Delete").build();
    let up = Button::builder(&side).with_label("Up").build();
    let down = Button::builder(&side).with_label("Down").build();
    let name = TextCtrl::builder(&side).build();
    let visible = CheckBox::builder(&side).with_label("Show in spectrogram").build();
    let action = Choice::builder(&side)
        .with_choices(vec!["Suppress".into(), "Extract".into()])
        .build();
    let parts = CheckListBox::builder(&side).with_size(Size::new(280, 90)).build();
    let reduction_label = StaticText::builder(&side).with_label("Max. reduction: 18 dB").build();
    let reduction = Slider::builder(&side)
        .with_value(18)
        .with_min_value(0)
        .with_max_value(48)
        .build();
    let learn = Button::builder(&side).with_label("Learn noise from selection").build();
    let adaptive = Button::builder(&side).with_label("Adaptive noise estimate").build();
    let sounds_label = StaticText::builder(&side)
        .with_label(&sounds_text(DiscoverParams::default().components))
        .build();
    let sounds = Slider::builder(&side)
        .with_value(DiscoverParams::default().components as i32)
        .with_min_value(1)
        .with_max_value(MAX_SOUNDS)
        .build();
    let group = CheckBox::builder(&side)
        .with_label("Group patterns by timing and direction")
        .build();
    group.set_value(DiscoverParams::default().group);
    let find_sounds = Button::builder(&side)
        .with_label("Find sounds (in the selection, if any)")
        .build();
    let keep = Button::builder(&side).with_label("Keep").build();
    let merge = Button::builder(&side).with_label("Merge with...").build();
    let split = Button::builder(&side).with_label("Split").build();
    let discard = Button::builder(&side).with_label("Discard suggestions").build();
    let legend = StaticText::builder(&side)
        .with_label("Blue: remaining signal\nGreen: your selection\nRed: found to belong to it, also elsewhere,\n  or a found sound you picked\nOrange: both\n\nDrag to select, click to set play head,\nright-drag to scrub, wheel zooms\n(Shift scrolls, Ctrl: frequency), Space plays,\nCtrl+2 / Ctrl+3: hear the picked sound\nalone / removed, Ctrl+1: everything.")
        .build();

    let s = BoxSizer::builder(Orientation::Vertical).build();
    let label = |text: &str| StaticText::builder(&side).with_label(text).build();
    s.add(
        &label("Filters (ticked = active, Space switches;\nCtrl/Shift+click picks several)"),
        0,
        SizerFlag::All,
        4,
    );
    s.add(
        &filters.ctrl(),
        1,
        SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right,
        4,
    );
    let row1 = BoxSizer::builder(Orientation::Horizontal).build();
    row1.add(&add_suppress, 1, SizerFlag::All, 2);
    row1.add(&add_extract, 1, SizerFlag::All, 2);
    s.add_sizer(&row1, 0, SizerFlag::Expand, 2);
    let row2 = BoxSizer::builder(Orientation::Horizontal).build();
    row2.add(&delete, 1, SizerFlag::All, 2);
    row2.add(&up, 1, SizerFlag::All, 2);
    row2.add(&down, 1, SizerFlag::All, 2);
    s.add_sizer(&row2, 0, SizerFlag::Expand, 2);
    s.add(&label("Name"), 0, SizerFlag::Left | SizerFlag::Top, 4);
    s.add(&name, 0, SizerFlag::Expand | SizerFlag::All, 4);
    s.add(&visible, 0, SizerFlag::All, 4);
    s.add(&action, 0, SizerFlag::Expand | SizerFlag::All, 4);
    s.add(
        &label("Parts (uncheck to disable temporarily)"),
        0,
        SizerFlag::Left | SizerFlag::Top,
        4,
    );
    s.add(&parts, 0, SizerFlag::Expand | SizerFlag::All, 4);
    s.add(&reduction_label, 0, SizerFlag::Left | SizerFlag::Top, 4);
    s.add(&reduction, 0, SizerFlag::Expand | SizerFlag::All, 4);
    s.add(&learn, 0, SizerFlag::Expand | SizerFlag::All, 2);
    s.add(&adaptive, 0, SizerFlag::Expand | SizerFlag::All, 2);
    s.add(
        &label("Sounds in the recording"),
        0,
        SizerFlag::Left | SizerFlag::Top,
        6,
    );
    s.add(&sounds_label, 0, SizerFlag::Left | SizerFlag::Top, 4);
    s.add(&sounds, 0, SizerFlag::Expand | SizerFlag::All, 4);
    s.add(&group, 0, SizerFlag::All, 4);
    s.add(&find_sounds, 0, SizerFlag::Expand | SizerFlag::All, 2);
    let row3 = BoxSizer::builder(Orientation::Horizontal).build();
    row3.add(&keep, 1, SizerFlag::All, 2);
    row3.add(&merge, 1, SizerFlag::All, 2);
    row3.add(&split, 1, SizerFlag::All, 2);
    s.add_sizer(&row3, 0, SizerFlag::Expand, 2);
    s.add(&discard, 0, SizerFlag::Expand | SizerFlag::All, 2);
    s.add(&legend, 0, SizerFlag::All, 6);
    side.set_sizer(s, true);

    let (detection_panel, detection_title, detection) = build_detection(&frame);

    let panes = Panes::new(
        &frame,
        &Contents {
            spectrogram: &canvas,
            filters: &side,
            detection: &detection_panel,
            overview: &overview,
            log: &log,
        },
    );

    Ui {
        frame,
        toolbar,
        panes,
        canvas,
        overview,
        detection_title,
        detection,
        filters,
        name,
        visible,
        action,
        parts,
        reduction,
        reduction_label,
        add_suppress,
        add_extract,
        delete,
        up,
        down,
        learn,
        adaptive,
        find_sounds,
        sounds,
        sounds_label,
        group,
        keep,
        merge,
        split,
        discard,
    }
}

/// Most spectral patterns the slider offers.
const MAX_SOUNDS: i32 = 16;

/// Label of the pattern-count slider.
fn sounds_text(n: usize) -> String {
    format!("Patterns to look for: {n}")
}

/// Builds the detection settings pane: a title naming the object being
/// edited, then one labelled slider per [`Param`], grouped into "parts of
/// this sound" and "repetitions elsewhere".
fn build_detection(frame: &Frame) -> (ScrolledWindow, StaticText, [(Slider, StaticText); Param::ALL.len()]) {
    let panel = ScrolledWindow::builder(frame).build();
    panel.set_scroll_rate(0, 10);
    let s = BoxSizer::builder(Orientation::Vertical).build();
    let title = StaticText::builder(&panel)
        .with_label("Mark a sound or pick a sound filter")
        .build();
    s.add(&title, 0, SizerFlag::All | SizerFlag::Expand, 6);
    let mut group = None;
    let controls = Param::ALL.map(|p| {
        if group != Some(p.group()) {
            group = Some(p.group());
            let heading = match p.group() {
                Group::Parts => "What else belongs to this sound",
                Group::Repetitions => "Where else it occurs",
            };
            let h = StaticText::builder(&panel).with_label(heading).build();
            s.add(&h, 0, SizerFlag::Left | SizerFlag::Top, 8);
        }
        // The name on its own line, below it the slider with its value.
        let name = StaticText::builder(&panel).with_label(p.label()).build();
        name.set_tooltip(p.help());
        s.add(&name, 0, SizerFlag::Left | SizerFlag::Right | SizerFlag::Top, 6);
        let row = BoxSizer::builder(Orientation::Horizontal).build();
        let slider = Slider::builder(&panel)
            .with_value(0)
            .with_min_value(0)
            .with_max_value(p.steps())
            .build();
        slider.set_tooltip(p.help());
        let value = StaticText::builder(&panel).with_label("").build();
        // Room for the longest value text, so it is not cut off later.
        value.set_min_size(Size::new(95, -1));
        row.add(&slider, 1, SizerFlag::AlignCenterVertical, 0);
        row.add(&value, 0, SizerFlag::AlignCenterVertical | SizerFlag::Left, 4);
        s.add_sizer(&row, 0, SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right, 4);
        (slider, value)
    });
    panel.set_sizer(s, true);
    (panel, title, controls)
}

/// Connects all widget events to the state.
fn bind_events(state: &Shared, ui: Ui) {
    // --- menus ---
    let st = state.clone();
    ui.frame.on_menu(move |e| match e.get_id() {
        ID_OPEN_AUDIO => {
            if let Some(path) = file_dialog(ui, "Open audio", &audio_wildcard(), false) {
                start_loading(&mut st.borrow_mut(), path);
            }
        }
        ID_OPEN_PROJECT => {
            if let Some(path) = file_dialog(ui, "Open project", PROJECT_WILDCARD, false) {
                open_project(&st, ui, &path);
            }
        }
        ID_SAVE_PROJECT => {
            if let Some(path) = file_dialog(ui, "Save project", PROJECT_WILDCARD, true) {
                let mut s = st.borrow_mut();
                match s.project.save(&path) {
                    Ok(()) => {
                        report(ui, &format!("Saved {}", path.display()));
                        s.remember(ui, RecentKind::Project, &path);
                        s.project_path = Some(path);
                    }
                    Err(err) => error_box(ui, &format!("{err:#}")),
                }
            }
        }
        ID_EXPORT => {
            if st.borrow().doc.is_none() {
                return;
            }
            if let Some(path) = file_dialog(ui, "Export processed audio", "WAV files|*.wav", true) {
                start_export(&mut st.borrow_mut(), path);
            }
        }
        ID_PLAY => toggle_play(&mut st.borrow_mut()),
        id if LISTEN_MENU.iter().any(|&(_, m)| m == id) => {
            let mode = LISTEN_MENU
                .iter()
                .find(|&&(_, m)| m == id)
                .map_or(Listen::All, |&(l, _)| l);
            set_listen(&mut st.borrow_mut(), ui, mode);
        }
        ID_ZOOM_FIT => {
            let mut s = st.borrow_mut();
            let width = canvas_size(ui).0;
            if let Some(vp) = s.viewport.as_mut() {
                let log = vp.view.log_freq;
                *vp = Viewport::fit(vp.total_samples, vp.sample_rate, width);
                if log {
                    vp.toggle_log();
                }
            }
            request_repaint();
        }
        ID_LOG_FREQ => {
            if let Some(vp) = st.borrow_mut().viewport.as_mut() {
                vp.toggle_log();
            }
            request_repaint();
        }
        ID_ADD_SUPPRESS => add_filter_from_selection(&st, ui, FilterAction::Suppress),
        ID_ADD_EXTRACT => add_filter_from_selection(&st, ui, FilterAction::Extract),
        ID_LEARN_NOISE => learn_noise(&st, ui),
        ID_FIND_SOUNDS => start_discovery(&mut st.borrow_mut(), ui),
        ID_QUIT => ui.frame.close(true),
        id => {
            if let Some(cmd) = RecentCommand::for_menu_id(id) {
                run_recent(&st, ui, cmd);
            } else if let Some(pane) = Pane::for_menu_id(id) {
                ui.panes.toggle(pane);
            } else {
                e.skip(true);
            }
        }
    });
    // Remember the layout for the next start; detach the panes before the
    // frame goes away, as wxWidgets asks.
    ui.frame.on_close(move |e| {
        ui.panes.save();
        ui.panes.uninit();
        e.skip(true);
    });

    // --- spectrogram canvas ---
    let st = state.clone();
    ui.canvas.on_paint(move |_| {
        let dc = AutoBufferedPaintDC::new(&ui.canvas);
        // GTK may deliver paint events synchronously from inside another
        // handler that is still mutating the state (e.g. while it updates
        // the status bar, or during a modal dialog). Painting then would
        // need a second borrow; instead remember to repaint on the next
        // timer tick.
        match st.try_borrow() {
            Ok(s) => paint(&s, &dc),
            Err(_) => REPAINT_PENDING.with(|p| p.set(true)),
        }
    });
    ui.canvas.on_size(move |_| ui.canvas.refresh(false, None));

    let st = state.clone();
    ui.canvas.on_mouse_left_down(move |e| {
        let Some(p) = mouse_pos(&e) else { return };
        ui.canvas.set_focus();
        ui.canvas.capture_mouse();
        if let Ok(mut s) = st.try_borrow_mut() {
            s.drag = Some((p, p));
        }
    });
    let st = state.clone();
    ui.canvas.on_mouse_motion(move |e| {
        let Some(p) = mouse_pos(&e) else { return };
        let Ok(mut s) = st.try_borrow_mut() else { return };
        if let Some((start, _)) = s.drag {
            s.drag = Some((start, p));
            request_repaint();
        }
        if s.scrub.is_some() {
            scrub_to(&mut s, p.0);
        }
        if let Some(vp) = &s.viewport {
            let (_, h) = canvas_size(ui);
            let t = vp.sample_at(p.0) as f64 / vp.sample_rate as f64;
            let hz = vp.view.y_to_hz(p.1, h);
            ui.frame
                .set_status_text(&format!("{}  {:.0} Hz", format_time(t), hz), 2);
        }
    });
    let st = state.clone();
    ui.canvas.on_mouse_left_up(move |e| {
        if ui.canvas.has_capture() {
            ui.canvas.release_mouse();
        }
        let p = mouse_pos(&e);
        let Ok(mut s) = st.try_borrow_mut() else { return };
        let Some((start, cur)) = s.drag.take() else { return };
        let end = p.unwrap_or(cur);
        finish_drag(&mut s, ui, start, end);
        request_repaint();
    });
    // Right button: scrub.
    let st = state.clone();
    ui.canvas.on_mouse_right_down(move |e| {
        let Some(p) = mouse_pos(&e) else { return };
        ui.canvas.capture_mouse();
        if let Ok(mut s) = st.try_borrow_mut() {
            begin_scrub(&mut s, p.0);
        }
    });
    let st = state.clone();
    ui.canvas.on_mouse_right_up(move |_| {
        if ui.canvas.has_capture() {
            ui.canvas.release_mouse();
        }
        if let Ok(mut s) = st.try_borrow_mut() {
            end_scrub(&mut s);
        }
    });
    let st = state.clone();
    ui.canvas.on_mouse_wheel(move |e| {
        let data = match &e {
            WindowEventData::MouseButton(m) => &m.event,
            WindowEventData::MouseMotion(m) => &m.event,
            _ => return,
        };
        let rotation = data.get_wheel_rotation();
        let x = data.get_position().map_or(0.0, |p| p.x as f32);
        let Ok(mut s) = st.try_borrow_mut() else { return };
        let (w, _) = canvas_size(ui);
        let Some(vp) = s.viewport.as_mut() else { return };
        let towards = if rotation > 0 { 1.0 } else { -1.0 };
        if get_key_state(WXK_SHIFT) {
            vp.scroll(-towards * w as f32 / 10.0, w);
        } else if get_key_state(WXK_CONTROL) {
            vp.zoom_freq(if towards > 0.0 { 0.8 } else { 1.25 });
        } else {
            vp.zoom_time(x, if towards > 0.0 { 0.8 } else { 1.25 }, w);
        }
        request_repaint();
    });
    let st = state.clone();
    ui.canvas.on_key_down(move |e| {
        let WindowEventData::Keyboard(k) = &e else { return };
        let Ok(mut s) = st.try_borrow_mut() else { return };
        match k.get_key_code() {
            Some(WXK_SPACE) => toggle_play(&mut s),
            Some(WXK_ESCAPE) => {
                s.selection = None;
                s.project.preview = None;
                push_project(&s);
                refresh_detection(&s, ui);
                request_repaint();
            }
            Some(WXK_HOME) => s.playhead = 0,
            _ => e.skip(true),
        }
    });

    // --- overview ---
    let st = state.clone();
    ui.overview.on_paint(move |_| {
        let dc = AutoBufferedPaintDC::new(&ui.overview);
        match st.try_borrow() {
            Ok(s) => paint_overview(&s, ui, &dc),
            Err(_) => REPAINT_PENDING.with(|p| p.set(true)),
        }
    });
    ui.overview.on_size(move |_| ui.overview.refresh(false, None));
    let st = state.clone();
    ui.overview.on_mouse_left_down(move |e| {
        let Some(p) = mouse_pos(&e) else { return };
        ui.overview.capture_mouse();
        let Ok(mut s) = st.try_borrow_mut() else { return };
        let (w, _) = canvas_size(ui);
        let (Some(vp), Some(o)) = (s.viewport, s.overview.as_mut()) else {
            return;
        };
        o.set_width(ui.overview.get_client_size().width.max(1) as u32);
        let start = o.press(p.0 as f64, vp.visible_range(w));
        scroll_to(&mut s, ui, start);
    });
    let st = state.clone();
    ui.overview.on_mouse_motion(move |e| {
        let Some(p) = mouse_pos(&e) else { return };
        let Ok(mut s) = st.try_borrow_mut() else { return };
        let Some(o) = s.overview.as_mut() else { return };
        if let Some(start) = o.drag(p.0 as f64) {
            scroll_to(&mut s, ui, start);
        } else if o.span().is_some() {
            // Only the rubber band changed.
            request_repaint();
        }
    });
    let st = state.clone();
    ui.overview.on_mouse_left_up(move |_| {
        if ui.overview.has_capture() {
            ui.overview.release_mouse();
        }
        let Ok(mut s) = st.try_borrow_mut() else { return };
        let Some(span) = s.overview.as_mut().and_then(|o| o.release()) else {
            // Also removes a dropped span's rubber band.
            request_repaint();
            return;
        };
        let (w, _) = canvas_size(ui);
        if let Some(vp) = s.viewport.as_mut() {
            vp.show_range(span.0, span.1, w);
        }
        request_repaint();
    });

    // --- detection settings ---
    for (i, &(slider, value)) in ui.detection.iter().enumerate() {
        let st = state.clone();
        let param = Param::ALL[i];
        slider.on_slider(move |e| {
            let v = param.value_at(e.get_value());
            value.set_label(&param.format(v));
            let Ok(mut s) = st.try_borrow_mut() else { return };
            set_detection_param(&mut s, param, v);
        });
    }

    // --- filter list and details ---
    let st = state.clone();
    ui.filters.on_pick(move |picked, clicked| {
        // Rebuilding the list selects rows, too; those events arrive while
        // the state is borrowed and are not the user's.
        let Ok(mut s) = st.try_borrow_mut() else { return };
        match clicked {
            Some(row) => {
                s.current_filter = Some(row);
                // Picking a filter means looking at it: a selection that
                // was not turned into a filter is dropped, and the picked
                // filter is drawn (see `push_project`).
                s.project.preview = None;
            }
            // A row was deselected: keep the details on a picked row.
            None if !s.current_filter.is_some_and(|c| picked.contains(&c)) => {
                if let Some(&last) = picked.last() {
                    s.current_filter = Some(last);
                }
            }
            None => {}
        }
        s.picked = picked;
        push_project(&s);
        refresh_details(&s, ui);
        refresh_detection(&s, ui);
    });
    let st = state.clone();
    ui.filters.on_toggle(move |rows| {
        let Ok(mut s) = st.try_borrow_mut() else { return };
        let rows: Vec<usize> = rows.into_iter().filter(|&i| i < s.project.filters.len()).collect();
        let states: Vec<bool> = rows.iter().map(|&i| s.project.filters[i].enabled).collect();
        let on = filter_list::toggled_state(&states);
        for &i in &rows {
            s.project.filters[i].enabled = on;
            ui.filters.set_enabled(i, on);
        }
        push_project(&s);
    });
    let st = state.clone();
    ui.name.on_text_updated(move |_| {
        let Ok(mut s) = st.try_borrow_mut() else { return };
        let Some(i) = s.current_filter else { return };
        let text = ui.name.get_value();
        if let Some(f) = s.project.filters.get_mut(i) {
            if f.name != text {
                f.name = text;
                refresh_filter_list(&s, ui);
            }
        }
    });
    let st = state.clone();
    ui.visible.on_toggled(move |_| {
        with_current_filter(&st, ui, |f| f.visible = ui.visible.is_checked());
    });
    let st = state.clone();
    ui.action.on_selection_changed(move |_| {
        let action = if ui.action.get_selection() == Some(1) {
            FilterAction::Extract
        } else {
            FilterAction::Suppress
        };
        with_current_filter(&st, ui, |f| f.action = action);
        refresh_filter_list(&st.borrow(), ui);
    });
    let st = state.clone();
    ui.parts.on_toggled(move |e| {
        let Some(i) = e.get_selection() else { return };
        let on = ui.parts.is_checked(i);
        with_current_filter(&st, ui, |f| {
            f.set_part_enabled(i as usize, on);
        });
        // A noise-reduction filter's single part cannot be switched; show
        // the real state again.
        refresh_details(&st.borrow(), ui);
    });
    let st = state.clone();
    ui.reduction.on_slider(move |e| {
        let db = e.get_value() as f32;
        ui.reduction_label.set_label(&format!("Max. reduction: {db:.0} dB"));
        with_current_filter(&st, ui, |f| {
            if let FilterKind::NoiseReduction(nr) = &mut f.kind {
                nr.reduction_db = db;
            }
        });
    });
    let st = state.clone();
    ui.add_suppress
        .on_click(move |_| add_filter_from_selection(&st, ui, FilterAction::Suppress));
    let st = state.clone();
    ui.add_extract
        .on_click(move |_| add_filter_from_selection(&st, ui, FilterAction::Extract));
    let st = state.clone();
    ui.delete.on_click(move |_| {
        let mut s = st.borrow_mut();
        let Some(i) = s.current_filter.filter(|&i| i < s.project.filters.len()) else {
            return;
        };
        s.project.filters.remove(i);
        s.project.prune_decompositions();
        let next = if s.project.filters.is_empty() {
            None
        } else {
            Some(i.min(s.project.filters.len() - 1))
        };
        s.focus_filter(next);
        push_project(&s);
        refresh_filter_list(&s, ui);
    });
    let st = state.clone();
    ui.up.on_click(move |_| move_filter(&st, ui, -1));
    let st = state.clone();
    ui.down.on_click(move |_| move_filter(&st, ui, 1));
    let st = state.clone();
    ui.learn.on_click(move |_| learn_noise(&st, ui));
    ui.sounds.on_slider(move |e| {
        ui.sounds_label.set_label(&sounds_text(e.get_value().max(1) as usize));
    });
    let st = state.clone();
    ui.find_sounds
        .on_click(move |_| start_discovery(&mut st.borrow_mut(), ui));
    let st = state.clone();
    ui.keep.on_click(move |_| {
        with_current_filter(&st, ui, |f| f.suggested = false);
        refresh_filter_list(&st.borrow(), ui);
    });
    let st = state.clone();
    ui.merge.on_click(move |_| merge_dialog(&st, ui));
    let st = state.clone();
    ui.split.on_click(move |_| {
        let mut s = st.borrow_mut();
        let Some(i) = s.current_filter else { return };
        match s.project.split(i) {
            Ok(n) => report(ui, &format!("Split into {n} sounds, one per pattern")),
            Err(e) => report(ui, &format!("{e:#}")),
        }
        push_project(&s);
        refresh_filter_list(&s, ui);
    });
    let st = state.clone();
    ui.discard.on_click(move |_| {
        let mut s = st.borrow_mut();
        let n = s.project.discard_suggestions();
        let len = s.project.filters.len();
        let next = s
            .current_filter
            .filter(|_| len > 0)
            .map(|i| i.min(len.saturating_sub(1)));
        s.focus_filter(next);
        push_project(&s);
        refresh_filter_list(&s, ui);
        report(ui, &format!("Discarded {n} suggestion(s)"));
    });
    let st = state.clone();
    ui.adaptive.on_click(move |_| {
        with_current_filter(&st, ui, |f| {
            if let FilterKind::NoiseReduction(nr) = &mut f.kind {
                nr.estimate = NoiseEstimate::default();
            }
        });
        report(ui, "Noise estimate: adaptive");
    });

    // --- background polling ---
    let timer = Timer::new(&ui.frame);
    let st = state.clone();
    timer.on_tick(move |_| {
        if let Ok(mut s) = st.try_borrow_mut() {
            tick(&mut s, ui);
        }

        // Outside the borrow, so the paint handler can read the state.
        if REPAINT_PENDING.with(|p| p.replace(false)) {
            ui.canvas.refresh(false, None);
            ui.overview.refresh(false, None);
        }
    });
    timer.start(40, false);
    // The timer must live as long as the frame; leaking the handle is the
    // simplest way to achieve that for the main window.
    std::mem::forget(timer);
}

/// Position of a mouse event in pixels.
fn mouse_pos(e: &WindowEventData) -> Option<(f32, f32)> {
    let p = match e {
        WindowEventData::MouseButton(m) => m.get_position(),
        WindowEventData::MouseMotion(m) => m.get_position(),
        _ => None,
    }?;
    Some((p.x as f32, p.y as f32))
}

/// Client size of the spectrogram canvas in pixels (at least 1×1).
fn canvas_size(ui: Ui) -> (u32, u32) {
    let s = ui.canvas.get_client_size();
    (s.width.max(1) as u32, s.height.max(1) as u32)
}

/// Shows an open or save dialog and returns the chosen path.
fn file_dialog(ui: Ui, title: &str, wildcard: &str, save: bool) -> Option<PathBuf> {
    let style = if save {
        FileDialogStyle::Save | FileDialogStyle::OverwritePrompt
    } else {
        FileDialogStyle::Open | FileDialogStyle::FileMustExist
    };
    let dlg = FileDialog::builder(&ui.frame)
        .with_message(title)
        .with_wildcard(wildcard)
        .with_style(style)
        .build();
    if dlg.show_modal() == ID_OK {
        dlg.get_path().map(PathBuf::from)
    } else {
        None
    }
}

/// Shows an error message.
fn error_box(ui: Ui, text: &str) {
    MessageDialog::builder(&ui.frame, text, "noiseextractor")
        .with_style(MessageDialogStyle::OK | MessageDialogStyle::IconError)
        .build()
        .show_modal();
}

/// Decodes `path` on a background thread; [`tick`] picks up the result.
fn start_loading(s: &mut State, path: PathBuf) {
    let tx = s.tx.clone();
    s.busy = Some(format!("Loading {} ...", path.display()));
    std::thread::spawn(move || {
        let result = AudioData::load(&path).map(|audio| {
            let peaks = Peaks::new(&audio.mono());
            (audio, peaks)
        });
        let _ = tx.send(Background::Loaded(path, result));
    });
}

/// Installs a freshly loaded recording: new engine, new player, fitted view.
fn install_audio(s: &mut State, ui: Ui, path: &Path, audio: AudioData, peaks: Peaks) {
    // Stop the old player before its engine goes away.
    s.doc = None;
    let audio = Arc::new(audio);
    s.project.preview = None;
    // Event filters from a project loaded earlier refer to this recording now.
    for f in &mut s.project.filters {
        if let FilterKind::Event(e) = &mut f.kind {
            e.analysis = None;
        }
    }
    s.project.prepare(&audio);
    let engine = Arc::new(Engine::new(audio.clone(), s.project.clone(), EngineConfig::default()));
    let player = match Player::new(engine.clone()) {
        Ok(p) => Some(p),
        Err(e) => {
            report(ui, &format!("Playback unavailable: {e:#}"));
            None
        }
    };
    let name = path
        .file_name()
        .map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
    ui.frame.set_title(&format!("{name} - noiseextractor"));
    let (w, _) = canvas_size(ui);
    s.viewport = Some(Viewport::fit(audio.len(), audio.sample_rate, w));
    s.overview = Some(Overview::new(
        audio.len(),
        ui.overview.get_client_size().width.max(1) as u32,
    ));
    s.selection = None;
    s.playhead = 0;
    s.scrub = None;
    report(
        ui,
        &format!(
            "Loaded {name}: {} channel(s), {} Hz, {}",
            audio.channels.len(),
            audio.sample_rate,
            format_time(audio.len() as f64 / audio.sample_rate as f64)
        ),
    );
    s.doc = Some(Doc {
        audio,
        engine,
        player,
        name,
        peaks,
    });
    s.last_revision = u64::MAX;
    refresh_detection(s, ui);
    ui.overview.refresh(false, None);
}

/// Loads a project file, keeping the current recording.
fn open_project(st: &Shared, ui: Ui, path: &Path) {
    let mut project = match Project::load(path) {
        Ok(p) => p,
        Err(e) => return error_box(ui, &format!("{e:#}")),
    };
    let mut s = st.borrow_mut();
    s.remember(ui, RecentKind::Project, path);
    s.project_path = Some(path.to_path_buf());
    s.focus_filter(if project.filters.is_empty() { None } else { Some(0) });
    if let Some(doc) = &s.doc {
        project.prepare(&doc.audio);
        if project.stft != doc.engine.project().stft {
            // A different STFT needs a new engine; reinstalling is simplest.
            let audio = (*doc.audio).clone();
            let name = doc.name.clone();
            let peaks = doc.peaks.clone();
            s.project = project;
            install_audio(&mut s, ui, Path::new(&name), audio, peaks);
        } else {
            s.project = project;
            push_project(&s);
        }
    } else {
        s.project = project;
    }
    refresh_filter_list(&s, ui);
}

/// Shows the recent-files list of `kind` in its submenu and stores it.
fn show_recent(ui: Ui, kind: RecentKind, list: &RecentFiles) {
    recent::save(kind, list);
    if let Some(bar) = ui.frame.get_menu_bar() {
        recent::refresh(&bar, kind, list);
    }
}

/// Runs a command of a recent-files submenu: opens the file, or takes it
/// off the list if it is gone (moved, deleted, on an unplugged drive).
fn run_recent(st: &Shared, ui: Ui, cmd: RecentCommand) {
    let (kind, index) = match cmd {
        RecentCommand::Clear(kind) => {
            let mut s = st.borrow_mut();
            let list = s.recent_mut(kind);
            list.clear();
            show_recent(ui, kind, list);
            return;
        }
        RecentCommand::Open(kind, index) => (kind, index),
    };
    let Some(path) = st.borrow_mut().recent_mut(kind).get(index).map(Path::to_path_buf) else {
        return;
    };
    if !path.is_file() {
        let mut s = st.borrow_mut();
        let list = s.recent_mut(kind);
        list.remove(&path);
        show_recent(ui, kind, list);
        report(
            ui,
            &format!("{} no longer exists; removed it from the list", path.display()),
        );
        return;
    }
    match kind {
        RecentKind::Audio => start_loading(&mut st.borrow_mut(), path),
        RecentKind::Project => open_project(st, ui, &path),
    }
}

/// Hands the current project to the engine, which recomputes in the
/// background (nearest to the focus first). The filter picked in the list
/// is drawn as if it were visible, so picking it shows what it does; while
/// a fresh selection is being edited, only that is highlighted.
///
/// In the solo and "without" listening modes the engine gets the project
/// derived for that mode instead ([`Project::for_listening`]), so what is
/// played and drawn in blue is the picked sound alone or the rest.
fn push_project(s: &State) {
    if let Some(doc) = &s.doc {
        let mut project = s.project.for_listening(s.listen, &s.listen_picks());
        // A selection that for_listening turned into a filter stays
        // highlighted, so the overview keeps showing what it finds.
        let moved_selection = s.project.preview.is_some() && project.preview.is_none();
        if !moved_selection {
            project.highlight = highlighted_filter(s.current_filter, s.project.preview.is_some());
        }
        doc.engine.set_project(project);
    }
}

/// The sounds the solo and "without" listening modes refer to: the
/// selection being edited if there is one (it is what the user just
/// marked), else the filters `picked` in the list.
fn listen_picks(picked: &[usize], editing_selection: bool) -> Vec<Pick> {
    if editing_selection {
        vec![Pick::Preview]
    } else {
        picked.iter().map(|&i| Pick::Filter(i)).collect()
    }
}

/// Switches the listening mode, keeps the View menu's radio items in
/// step (the mode may also be chosen elsewhere, e.g. a toolbar) and
/// reprocesses; playback goes on and is heard in the new mode as soon as
/// the chunks around the play head are done.
fn set_listen(s: &mut State, ui: Ui, mode: Listen) {
    s.listen = mode;
    if let Some(bar) = ui.frame.get_menu_bar() {
        for (l, id) in LISTEN_MENU {
            bar.check_item(id, l == mode);
        }
    }
    if let Some(tb) = ui.toolbar {
        for (l, id) in LISTEN_MENU {
            tb.toggle_tool(id, l == mode);
        }
    }
    let n = s.listen_picks().len();
    let what = match (mode, n) {
        (Listen::All, _) => "Hearing everything".to_string(),
        (_, 0) => "Mark a sound or pick filters to hear them alone or without them".to_string(),
        (Listen::Solo, n) => format!(
            "Hearing only the {n} picked sound(s) (follows the selection, or the filters picked with Ctrl/Shift+click)"
        ),
        (Listen::Without, n) => format!(
            "Hearing everything but the {n} picked sound(s) (follows the selection, or the filters picked with \
             Ctrl/Shift+click)"
        ),
    };
    report(ui, &what);
    push_project(s);
}

/// The filter whose overlay is drawn although it may not be marked visible
/// (see [`push_project`]): the current one, unless a fresh selection is
/// being edited, whose colours would be mixed up with it.
fn highlighted_filter(current: Option<usize>, editing_selection: bool) -> Option<usize> {
    current.filter(|_| !editing_selection)
}

/// Applies `change` to the filter shown in the details and pushes the
/// project.
fn with_current_filter(st: &Shared, _ui: Ui, change: impl FnOnce(&mut Filter)) {
    let Ok(mut s) = st.try_borrow_mut() else { return };
    let Some(i) = s.current_filter else { return };
    let Some(f) = s.project.filters.get_mut(i) else { return };
    change(f);
    push_project(&s);
}

/// Ends a mouse drag: a click moves the play head, a drag marks a
/// selection and starts its analysis.
fn finish_drag(s: &mut State, ui: Ui, start: (f32, f32), end: (f32, f32)) {
    let Some(vp) = s.viewport else { return };
    let (_, h) = canvas_size(ui);
    if (start.0 - end.0).abs() < 3.0 && (start.1 - end.1).abs() < 3.0 {
        // A click: move the play head (and keep playing from there).
        s.playhead = vp.sample_at(end.0);
        if let Some(p) = s.doc.as_ref().and_then(|d| d.player.as_ref()) {
            if p.is_playing() {
                p.play(s.playhead, None);
            }
        }
        return;
    }
    let sel = vp.drag_to_selection(start, end, h);
    s.selection = Some(sel);
    // Show the rectangle immediately; the analysis fills in the masks. The
    // detection settings last chosen carry over to the new selection.
    let mut preview = Filter {
        enabled: false,
        ..Filter::event("Selection", sel, FilterAction::Suppress)
    };
    if let FilterKind::Event(e) = &mut preview.kind {
        (e.params, e.similar) = s.last_params.clone();
    }
    s.project.preview = Some(preview);
    s.reanalyse_at = None;
    // Stop highlighting the picked filter right away; the selection's own
    // colours follow when its analysis is done.
    push_project(s);
    start_analysis(s, Target::Preview);
    refresh_detection(s, ui);
}

/// The object the detection sliders edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    /// The fresh selection, not yet a filter.
    Preview,
    /// Event filter number `.0` of the list.
    Filter(usize),
}

/// What the detection sliders edit: a fresh selection if there is one,
/// else the current filter if it is an event filter.
fn detection_target(s: &State) -> Option<Target> {
    if s.project.preview.is_some() {
        return Some(Target::Preview);
    }
    let i = s.current_filter?;
    matches!(s.project.filters.get(i)?.kind, FilterKind::Event(_)).then_some(Target::Filter(i))
}

/// The event filter behind `target`.
fn target_filter(s: &mut State, target: Target) -> Option<&mut EventFilter> {
    let f = match target {
        Target::Preview => s.project.preview.as_mut()?,
        Target::Filter(i) => s.project.filters.get_mut(i)?,
    };
    match &mut f.kind {
        FilterKind::Event(e) => Some(e),
        FilterKind::NoiseReduction(_) | FilterKind::Discovered(_) => None,
    }
}

/// Analyses the selection of `target` with its current settings on a
/// background thread; [`tick`] hands the result to every object that still
/// matches. The old analysis stays in use until then.
fn start_analysis(s: &mut State, target: Target) {
    let Some(doc) = &s.doc else { return };
    let mono = doc.engine.mono().clone();
    // The individual tracks tell where the marked sound comes from.
    let audio = doc.audio.clone();
    let sr = doc.audio.sample_rate;
    let cfg = s.project.stft;
    let noise = s.project.analysis_noise();
    let tx = s.tx.clone();
    let Some(e) = target_filter(s, target) else { return };
    let (sel, params) = (e.selection, e.params.clone());
    std::thread::spawn(move || {
        let a = analyze_selection(&mono, &audio.channels, sr, &Stft::new(cfg), &sel, &noise, &params);
        let _ = tx.send(Background::Analysed(sel, params, Arc::new(a)));
    });
}

/// One line about an analysis for the status bar and the log. `cfg` and
/// `sample_rate` convert bins and frames to Hz.
fn describe_analysis(a: &RegionAnalysis, cfg: StftConfig, sample_rate: u32) -> String {
    let found: Vec<String> = a.partials.iter().map(|p| format!("{:.2}", p.ratio)).collect();
    let mut text = if let Some((lo, hi)) = a.broadband {
        // A noise band has no pitch, so there are no partials to report.
        format!(
            "Broadband sound, {:.1} .. {:.1} kHz",
            cfg.bin_to_hz(lo as f32, sample_rate) / 1000.0,
            cfg.bin_to_hz(hi as f32, sample_rate) / 1000.0
        )
    } else if found.is_empty() {
        "No related partials found".to_string()
    } else {
        format!("Related partials at {} x the tracked pitch", found.join(", "))
    };
    if let Some(r) = a.template.as_ref().and_then(|t| t.rhythm.as_ref()) {
        let hz = sample_rate as f32 / (cfg.hop as f32 * r.period);
        text += &format!("; pulsed at {hz:.1} Hz");
    }
    if let Some(sig) = a.template.as_ref().and_then(|t| t.spatial.as_ref()) {
        for (i, p) in sig.pairs.iter().enumerate() {
            text += &format!(
                "; channel {} vs 0: {:+.1} dB, {:+.2} samples",
                i + 1,
                p.level_db,
                p.delay_samples
            );
        }
    }
    text
}

/// A detection slider moved: stores the value in the edited object (and
/// as the default for new selections). Repetition settings take effect
/// with the next processing; part settings need a new analysis, which
/// starts once the slider rests for [`REANALYSE_DELAY`].
fn set_detection_param(s: &mut State, param: Param, v: f32) {
    let Some(target) = detection_target(s) else { return };
    let Some(e) = target_filter(s, target) else { return };
    param.set(e, v);
    let params = (e.params.clone(), e.similar.clone());
    s.last_params = params;
    if param.reanalyse() {
        s.reanalyse_at = Some(Instant::now() + REANALYSE_DELAY);
    } else {
        push_project(s);
    }
}

/// Shows the edited object's detection settings in the sliders, and which
/// object that is. Settings that need several channels are disabled for
/// mono recordings.
fn refresh_detection(s: &State, ui: Ui) {
    let target = detection_target(s);
    let e = target.and_then(|t| match t {
        Target::Preview => s.project.preview.as_ref(),
        Target::Filter(i) => s.project.filters.get(i),
    });
    let e = e.and_then(|f| match &f.kind {
        FilterKind::Event(e) => Some((f.name.as_str(), e)),
        FilterKind::NoiseReduction(_) | FilterKind::Discovered(_) => None,
    });
    let title = match (target, e) {
        (Some(Target::Preview), Some(_)) => "Settings of the selection (not yet a filter)".to_string(),
        (_, Some((name, _))) => format!("Settings of \"{name}\""),
        _ => "Mark a sound or pick a sound filter".to_string(),
    };
    ui.detection_title.set_label(&title);
    let stereo = s.doc.as_ref().is_some_and(|d| d.audio.channels.len() >= 2);
    for (p, &(slider, value)) in Param::ALL.iter().zip(&ui.detection) {
        match e {
            Some((_, e)) => {
                let v = p.get(e);
                slider.set_value(p.slider_pos(v));
                value.set_label(&p.format(v));
                slider.enable(stereo || !p.spatial());
            }
            None => {
                value.set_label("");
                slider.enable(false);
            }
        }
    }
}

/// Turns the analysed selection into a new filter.
fn add_filter_from_selection(st: &Shared, ui: Ui, action: FilterAction) {
    let mut s = st.borrow_mut();
    let Some(preview) = s.project.preview.take() else {
        report(ui, "Drag a rectangle over the noise first");
        return;
    };
    let name = format!(
        "{} {}",
        if action == FilterAction::Extract {
            "Sound"
        } else {
            "Noise"
        },
        s.filter_counter
    );
    s.filter_counter += 1;
    s.project.filters.push(Filter {
        name,
        enabled: true,
        visible: true,
        action,
        ..preview
    });
    let last = s.project.filters.len() - 1;
    s.focus_filter(Some(last));
    s.selection = None;
    push_project(&s);
    refresh_filter_list(&s, ui);
}

/// Looks for the recording's sounds on a background thread, in the selected
/// time range if there is a selection; [`tick`] adds the result.
fn start_discovery(s: &mut State, ui: Ui) {
    let Some(doc) = &s.doc else {
        report(ui, "Open an audio file first");
        return;
    };
    if s.busy.is_some() {
        report(ui, "Busy, try again in a moment");
        return;
    }
    let params = DiscoverParams {
        components: ui.sounds.get_value().clamp(1, MAX_SOUNDS) as usize,
        group: ui.group.get_value(),
        range_s: s.selection.map(|sel| (sel.start_s, sel.end_s)),
        ..DiscoverParams::default()
    };
    let audio = doc.audio.clone();
    let cfg = s.project.stft;
    let noise = s.project.analysis_noise();
    let tx = s.tx.clone();
    s.busy = Some("Looking for sounds ...".into());
    report(
        ui,
        &format!(
            "Looking for {} patterns in {} ...",
            params.components,
            if params.range_s.is_some() {
                "the selected stretch"
            } else {
                "the recording"
            }
        ),
    );
    std::thread::spawn(move || {
        let _ = tx.send(Background::Discovered(discover(&audio, cfg, &noise, &params)));
    });
}

/// Asks which found sound to merge into the current one, then merges.
fn merge_dialog(st: &Shared, ui: Ui) {
    // Collect the choices first: no borrow may be held across the dialog.
    let (current, candidates, names) = {
        let s = st.borrow();
        let Some(i) = s.current_filter else { return };
        let candidates = s.project.merge_candidates(i);
        let names: Vec<String> = candidates.iter().map(|&j| s.project.filters[j].name.clone()).collect();
        (i, candidates, names)
    };
    if candidates.is_empty() {
        report(ui, "No other sound from the same search to merge with");
        return;
    }
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let dlg = SingleChoiceDialog::builder(&ui.frame, "Merge which sound into this one?", "Merge sounds", &refs).build();
    if dlg.show_modal() != ID_OK {
        return;
    }
    let Some(&from) = usize::try_from(dlg.get_selection())
        .ok()
        .and_then(|k| candidates.get(k))
    else {
        return;
    };
    let mut s = st.borrow_mut();
    match s.project.merge(current, from) {
        Ok(i) => {
            s.focus_filter(Some(i));
            report(ui, "Merged");
        }
        Err(e) => report(ui, &format!("{e:#}")),
    }
    push_project(&s);
    refresh_filter_list(&s, ui);
}

/// Moves the current filter up (`-1`) or down (`+1`) in the list.
fn move_filter(st: &Shared, ui: Ui, delta: i32) {
    let mut s = st.borrow_mut();
    let Some(i) = s.current_filter else { return };
    let j = i as i32 + delta;
    if j < 0 || j as usize >= s.project.filters.len() {
        return;
    }
    s.project.filters.swap(i, j as usize);
    s.focus_filter(Some(j as usize));
    push_project(&s);
    refresh_filter_list(&s, ui);
}

/// Sets the noise profile of the current (or first) noise-reduction filter
/// to the mean spectrum of the selected time range.
fn learn_noise(st: &Shared, ui: Ui) {
    let mut s = st.borrow_mut();
    let (Some(sel), Some(doc)) = (s.selection, s.doc.as_ref()) else {
        report(ui, "Select a stretch that contains only background noise first");
        return;
    };
    let cfg = s.project.stft;
    let (a, b) = sel.frames(cfg, doc.audio.sample_rate);
    let power = Stft::new(cfg).analyze(doc.engine.mono(), a, (b - a) as usize).power();
    let profile = mean_power(&power, cfg.bins());
    let target = s
        .current_filter
        .filter(|&i| {
            matches!(
                s.project.filters.get(i).map(|f| &f.kind),
                Some(FilterKind::NoiseReduction(_))
            )
        })
        .or_else(|| {
            s.project
                .filters
                .iter()
                .position(|f| matches!(f.kind, FilterKind::NoiseReduction(_)))
        });
    let i = match target {
        Some(i) => i,
        None => {
            s.project.filters.insert(0, Filter::noise_reduction("Background noise"));
            0
        }
    };
    if let FilterKind::NoiseReduction(nr) = &mut s.project.filters[i].kind {
        nr.estimate = NoiseEstimate::Profile(profile);
    }
    s.focus_filter(Some(i));
    push_project(&s);
    refresh_filter_list(&s, ui);
    report(ui, "Noise profile learned from the selection");
}

/// Starts or stops playback: the selection if there is one, otherwise from
/// the play head to the end.
fn toggle_play(s: &mut State) {
    let Some(doc) = &s.doc else { return };
    let Some(player) = &doc.player else { return };
    if player.is_playing() {
        player.pause();
        s.playhead = player.position();
        return;
    }
    match s.selection {
        Some(sel) => {
            let sr = doc.audio.sample_rate as f64;
            player.play((sel.start_s * sr) as usize, Some((sel.end_s * sr) as usize));
        }
        None => player.play(s.playhead, None),
    }
}

/// Exports the processed recording on a background thread.
fn start_export(s: &mut State, path: PathBuf) {
    let Some(doc) = &s.doc else { return };
    let audio = doc.audio.clone();
    let project = s.project.clone();
    let tx = s.tx.clone();
    s.busy = Some("Exporting ...".into());
    std::thread::spawn(move || {
        let run = || -> anyhow::Result<PathBuf> {
            let mut w = WavStreamWriter::create(&path, audio.sample_rate, audio.channels.len() as u16)?;
            let total = audio.len().max(1);
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            process_all(&audio, &project, EngineConfig::default().chunk_frames, threads, |r| {
                let _ = tx.send(Background::ExportProgress(
                    (r.start_sample + r.audio[0].len()) * 100 / total,
                ));
                w.write(&r.audio)
            })?;
            w.finish()?;
            Ok(path.clone())
        };
        let _ = tx.send(Background::Exported(run()));
    });
}

/// Periodic work on the GUI thread: collect background results, update the
/// engine's focus, repaint when needed and show progress.
fn tick(s: &mut State, ui: Ui) {
    while let Ok(msg) = s.rx.try_recv() {
        match msg {
            Background::Loaded(path, Ok((audio, peaks))) => {
                s.busy = None;
                s.remember(ui, RecentKind::Audio, &path);
                install_audio(s, ui, &path, audio, peaks);
            }
            Background::Loaded(path, Err(e)) => {
                s.busy = None;
                report(ui, &format!("Cannot open {}: {e:#}", path.display()));
            }
            Background::Analysed(sel, params, a) => {
                // Every object that still has this selection and these
                // settings takes the result; one that changed since waits
                // for its own, newer analysis.
                let mut taken = false;
                for f in s.project.filters.iter_mut().chain(s.project.preview.as_mut()) {
                    if let FilterKind::Event(e) = &mut f.kind {
                        if e.selection == sel && e.params == params {
                            e.analysis = Some(a.clone());
                            taken = true;
                        }
                    }
                }
                if taken {
                    let sr = s.doc.as_ref().map_or(1, |d| d.audio.sample_rate);
                    report(ui, &describe_analysis(&a, s.project.stft, sr));
                    push_project(s);
                }
            }
            Background::Discovered(result) => {
                s.busy = None;
                match result {
                    Ok(found) => {
                        let (patterns, frames) = (found.decomposition.patterns.len(), found.frames_used);
                        let sounds = found.groups.len();
                        let first = s.project.add_discovery(found);
                        s.focus_filter(Some(first));
                        push_project(s);
                        refresh_filter_list(s, ui);
                        report(
                            ui,
                            &format!(
                                "Found {sounds} sound(s) from {patterns} patterns ({frames} frames): pick one to see it in \
                                 red, check it to apply it; Keep, Merge, Split or Discard"
                            ),
                        );
                    }
                    Err(e) => report(ui, &format!("Finding sounds failed: {e:#}")),
                }
            }
            Background::ExportProgress(p) => s.busy = Some(format!("Exporting ... {p}%")),
            Background::Exported(r) => {
                s.busy = None;
                report(
                    ui,
                    &match r {
                        Ok(p) => format!("Exported {}", p.display()),
                        Err(e) => format!("Export failed: {e:#}"),
                    },
                );
            }
        }
    }

    let (w, _) = canvas_size(ui);
    let Some(doc) = &s.doc else {
        // Progress goes to the status bar only; the log gets the result.
        if let Some(b) = &s.busy {
            ui.frame.set_status_text(b, 0);
        }
        return;
    };
    if let Some(pos) = s.scrub.as_mut().and_then(|sc| sc.scrubber.tick(Instant::now())) {
        if let Some(p) = &doc.player {
            p.play(pos, None);
        }
    }
    if s.reanalyse_at.is_some_and(|t| Instant::now() >= t) {
        s.reanalyse_at = None;
        if let Some(target) = detection_target(s) {
            report(ui, "Analysing the selected sound again with the new settings ...");
            start_analysis(s, target);
        }
    }
    let Some(doc) = &s.doc else { return };
    let playing = doc.player.as_ref().is_some_and(|p| p.is_playing());
    if playing {
        s.playhead = doc.player.as_ref().map_or(s.playhead, |p| p.position());
    }
    let sr = doc.audio.sample_rate as f64;
    doc.engine.set_focus(Focus {
        playhead: Some(s.playhead),
        selection: s
            .selection
            .map(|sel| ((sel.start_s * sr) as usize, (sel.end_s * sr) as usize)),
        view: s.viewport.map(|vp| vp.visible_range(w)),
    });
    let rev = doc.engine.revision();
    if rev != s.last_revision || playing {
        s.last_revision = rev;
        request_repaint();
    }
    let (current, _, total) = doc.engine.progress();
    let progress = if current < total {
        format!("Processing {current}/{total} chunks")
    } else {
        "All chunks up to date".into()
    };
    ui.frame.set_status_text(s.busy.as_deref().unwrap_or(&progress), 1);
}

/// Draws the spectrogram, the selection, the drag rectangle, the play head
/// and the axis labels.
fn paint(s: &State, dc: &AutoBufferedPaintDC) {
    let (w, h) = dc.get_size();
    let (w, h) = (w.max(1) as u32, h.max(1) as u32);
    dc.set_background(Colour::rgb(20, 20, 26));
    dc.clear();
    let (Some(doc), Some(vp)) = (&s.doc, &s.viewport) else {
        dc.set_text_foreground(Colour::rgb(200, 200, 200));
        dc.draw_text(s.busy.as_deref().unwrap_or("File > Open audio... (Ctrl+O)"), 20, 20);
        return;
    };
    // Recompute the view for the current width (the window may have been
    // resized since the last event).
    let mut vp = *vp;
    vp.clamp(w);
    let engine = &doc.engine;
    let rgb = render_rgb(
        &vp.view,
        w,
        h,
        engine.layout(),
        doc.audio.sample_rate,
        s.project.stft.fft_size,
        &|c| engine.chunk(c),
    );
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.chunks(3) {
        rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    if let Some(bmp) = Bitmap::from_rgba(&rgba, w, h) {
        dc.draw_bitmap(&bmp, 0, 0, false);
    }

    dc.set_brush(Colour::rgb(0, 0, 0), BrushStyle::Transparent);
    if let Some(sel) = &s.selection {
        let (x, y, rw, rh) = vp.selection_rect(sel, h);
        dc.set_pen(Colour::rgb(120, 255, 120), 1, PenStyle::ShortDash);
        dc.draw_rectangle(x, y, rw, rh);
    }
    if let Some(((x0, y0), (x1, y1))) = s.drag {
        dc.set_pen(Colour::rgb(255, 255, 255), 1, PenStyle::Dot);
        dc.draw_rectangle(
            x0.min(x1) as i32,
            y0.min(y1) as i32,
            (x1 - x0).abs() as i32,
            (y1 - y0).abs() as i32,
        );
    }
    let px = vp.view.sample_to_x(s.playhead as f64);
    if (0.0..w as f32).contains(&px) {
        dc.set_pen(Colour::rgb(255, 230, 80), 1, PenStyle::Solid);
        dc.draw_line(px as i32, 0, px as i32, h as i32);
    }

    // Axis labels: a few frequencies on the left, times at the bottom.
    dc.set_text_foreground(Colour::rgb(220, 220, 220));
    for i in 1..6 {
        let y = h as f32 * i as f32 / 6.0;
        let hz = vp.view.y_to_hz(y, h);
        dc.draw_text(&format!("{hz:.0} Hz"), 4, y as i32 - 7);
    }
    for i in 0..6 {
        let x = w as f32 * i as f32 / 6.0;
        let t = vp.view.x_to_sample(x) / doc.audio.sample_rate as f64;
        dc.draw_text(&format_time(t), x as i32 + 4, h as i32 - 18);
    }
}

/// Rebuilds the filter list from the project and updates the details.
fn refresh_filter_list(s: &State, ui: Ui) {
    let rows: Vec<(String, bool)> = s
        .project
        .filters
        .iter()
        .map(|f| {
            let kind = match (&f.kind, f.action) {
                (FilterKind::NoiseReduction(_), FilterAction::Suppress) => "denoise",
                (FilterKind::NoiseReduction(_), FilterAction::Extract) => "keep noise",
                (FilterKind::Event(_), FilterAction::Suppress) => "suppress",
                (FilterKind::Event(_), FilterAction::Extract) => "extract",
                (FilterKind::Discovered(_), FilterAction::Suppress) => "found, suppress",
                (FilterKind::Discovered(_), FilterAction::Extract) => "found, extract",
            };
            let note = if f.suggested { " (suggested)" } else { "" };
            (format!("{} [{kind}]{note}", f.name), f.enabled)
        })
        .collect();
    let picked = filter_list::effective_picks(&s.picked, s.current_filter);
    ui.filters.set_rows(&rows, &picked, s.current_filter);
    refresh_details(s, ui);
}

/// Shows the current filter's name, visibility, action, parts and noise
/// settings; disables controls that do not apply.
fn refresh_details(s: &State, ui: Ui) {
    let f = s.current_filter.and_then(|i| s.project.filters.get(i));
    let has = f.is_some();
    ui.name.enable(has);
    ui.visible.enable(has);
    ui.action.enable(has);
    ui.parts.enable(has);
    ui.delete.enable(has);
    ui.parts.clear();
    ui.discard.enable(s.project.filters.iter().any(|f| f.suggested));
    let Some(f) = f else {
        ui.keep.enable(false);
        ui.merge.enable(false);
        ui.split.enable(false);
        ui.name.change_value("");
        ui.reduction.enable(false);
        refresh_detection(s, ui);
        return;
    };
    // change_value does not emit a text event, so no feedback loop.
    if ui.name.get_value() != f.name {
        ui.name.change_value(&f.name);
    }
    ui.visible.set_value(f.visible);
    ui.action
        .set_selection(if f.action == FilterAction::Extract { 1 } else { 0 });
    for (i, (label, on)) in f.parts().into_iter().enumerate() {
        ui.parts.append(&label);
        ui.parts.check(i as u32, on);
    }
    refresh_detection(s, ui);
    let found = matches!(f.kind, FilterKind::Discovered(_));
    ui.keep.enable(f.suggested);
    ui.merge.enable(found);
    ui.split.enable(found);
    let nr = matches!(f.kind, FilterKind::NoiseReduction(_));
    ui.reduction.enable(nr);
    ui.learn.enable(true);
    ui.adaptive.enable(nr);
    if let FilterKind::NoiseReduction(n) = &f.kind {
        ui.reduction.set_value(n.reduction_db.round() as i32);
        ui.reduction_label
            .set_label(&format!("Max. reduction: {:.0} dB", n.reduction_db));
    }
}

/// Shows `msg` in the status bar and keeps it in the log.
fn report(ui: Ui, msg: &str) {
    ui.frame.set_status_text(msg, 0);
    log::append(msg);
}

/// Right button down at pixel column `x`: starts scrubbing there.
fn begin_scrub(s: &mut State, x: f32) {
    let (Some(vp), Some(doc)) = (s.viewport, &s.doc) else {
        return;
    };
    let Some(player) = &doc.player else { return };
    let mut scrubber = Scrubber::default();
    let was_playing = player.is_playing();
    if let Some(pos) = scrubber.move_to(vp.sample_at(x), Instant::now()) {
        player.play(pos, None);
    }
    s.scrub = Some(Scrub { scrubber, was_playing });
}

/// The pointer moved to pixel column `x` while scrubbing.
fn scrub_to(s: &mut State, x: f32) {
    let (Some(vp), Some(doc), Some(scrub)) = (s.viewport, &s.doc, &mut s.scrub) else {
        return;
    };
    if let (Some(pos), Some(p)) = (scrub.scrubber.move_to(vp.sample_at(x), Instant::now()), &doc.player) {
        p.play(pos, None);
    }
}

/// Right button up: playback goes on if it ran before the scrub, else it
/// stops and the play head stays where the scrub ended.
fn end_scrub(s: &mut State) {
    let Some(scrub) = s.scrub.take() else { return };
    let Some(p) = s.doc.as_ref().and_then(|d| d.player.as_ref()) else {
        return;
    };
    if !scrub.was_playing {
        p.pause();
        s.playhead = scrub.scrubber.position().unwrap_or_else(|| p.position());
    }
    request_repaint();
}

/// Scrolls the spectrogram so that it starts at sample `start` (clamped).
fn scroll_to(s: &mut State, ui: Ui, start: usize) {
    let (w, _) = canvas_size(ui);
    if let Some(vp) = s.viewport.as_mut() {
        vp.view.start_sample = start as f64;
        vp.clamp(w);
    }
    request_repaint();
}

/// Draws the overview: the waveform of the whole recording, how much the
/// filter being worked on found where (orange), the selection's
/// time span, the frame around the part shown in the spectrogram, a span
/// being dragged out to zoom to, and the play head.
fn paint_overview(s: &State, ui: Ui, dc: &AutoBufferedPaintDC) {
    let (w, h) = dc.get_size();
    dc.set_background(Colour::rgb(20, 20, 26));
    dc.clear();
    let (Some(doc), Some(vp), Some(o)) = (&s.doc, &s.viewport, &s.overview) else {
        return;
    };
    let mut o = *o;
    o.set_width(w.max(1) as u32);
    let mid = h as f32 / 2.0;
    let scale = mid * 0.95;
    dc.set_pen(Colour::rgb(90, 140, 230), 1, PenStyle::Solid);
    for (x, (lo, hi)) in doc.peaks.columns(w.max(1) as u32).into_iter().enumerate() {
        let (y0, y1) = ((mid - hi * scale) as i32, (mid - lo * scale) as i32);
        dc.draw_line(x as i32, y0, x as i32, y1.max(y0 + 1));
    }
    // How much the filter being worked on found where: an orange envelope
    // on the waveform's scale, so its share of the recording shows.
    let [r, g, b] = ORANGE.map(|v| v as u8);
    dc.set_pen(Colour::rgb(r, g, b), 1, PenStyle::Solid);
    let engine = &doc.engine;
    for (x, a) in found_columns(w.max(1) as u32, engine.layout(), &|c| engine.chunk(c))
        .into_iter()
        .enumerate()
    {
        let Some(a) = a else { continue };
        let a = (a * scale).min(mid);
        if a >= 0.5 {
            dc.draw_line(x as i32, (mid - a) as i32, x as i32, (mid + a) as i32);
        }
    }
    let sr = doc.audio.sample_rate as f64;
    dc.set_brush(Colour::rgb(0, 0, 0), BrushStyle::Transparent);
    if let Some(sel) = &s.selection {
        let x0 = o.sample_to_x((sel.start_s * sr) as usize);
        let x1 = o.sample_to_x((sel.end_s * sr) as usize);
        dc.set_pen(Colour::rgb(120, 255, 120), 1, PenStyle::ShortDash);
        dc.draw_rectangle(x0 as i32, 1, (x1 - x0).max(1.0) as i32, h - 2);
    }
    // Kept inside the strip, so its edges show even when all is visible.
    let (l, r) = o.frame(vp.visible_range(canvas_size(ui).0));
    let (l, r) = (l.max(0.0), r.min(w as f64 - 1.0));
    dc.set_pen(Colour::rgb(240, 240, 240), 1, PenStyle::Solid);
    dc.draw_rectangle(l as i32, 0, (r - l).max(1.0) as i32, h);
    // The span being dragged out to zoom to.
    if let Some((l, r)) = o.span() {
        dc.set_pen(Colour::rgb(240, 240, 240), 1, PenStyle::ShortDash);
        dc.draw_rectangle(l as i32, 1, (r - l).max(1.0) as i32, h - 2);
    }
    let px = o.sample_to_x(s.playhead) as i32;
    dc.set_pen(Colour::rgb(255, 230, 80), 1, PenStyle::Solid);
    dc.draw_line(px, 0, px, h);
}

/// The Windows resource script that `build.rs` compiles, included here so
/// that its unit tests run on every platform.
#[cfg(test)]
#[path = "../build/resources.rs"]
mod resources;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_selection_being_edited_is_heard_before_the_picked_filters() {
        assert_eq!(listen_picks(&[2, 4], true), vec![Pick::Preview]);
        assert_eq!(listen_picks(&[2, 4], false), vec![Pick::Filter(2), Pick::Filter(4)]);
        assert!(listen_picks(&[], false).is_empty());
        assert_eq!(listen_picks(&[], true), vec![Pick::Preview]);
    }

    #[test]
    fn listening_modes_have_distinct_menu_ids() {
        for (i, (_, a)) in LISTEN_MENU.iter().enumerate() {
            assert!(LISTEN_MENU[i + 1..].iter().all(|(_, b)| a != b));
            assert!(![
                ID_OPEN_AUDIO,
                ID_OPEN_PROJECT,
                ID_SAVE_PROJECT,
                ID_EXPORT,
                ID_PLAY,
                ID_ZOOM_FIT,
                ID_LOG_FREQ,
                ID_QUIT
            ]
            .contains(a));
        }
    }

    #[test]
    fn the_picked_filter_is_highlighted_unless_a_selection_is_edited() {
        assert_eq!(highlighted_filter(Some(2), false), Some(2));
        assert_eq!(highlighted_filter(Some(2), true), None);
        assert_eq!(highlighted_filter(None, false), None);
    }
}
