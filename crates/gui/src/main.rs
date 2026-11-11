//! noiseextractor GUI: an interactive spectrogram in which noises can be
//! marked, inspected and suppressed or extracted with a list of filters.
//!
//! Layout: the spectrogram fills the window; the filter list ("building
//! kit") sits on the right. All heavy lifting happens in
//! `noiseextractor-core` on background threads; this crate only wires the
//! widgets to it. A 40 ms timer polls the background work (file loading,
//! selection analysis, chunk processing, playback position) and repaints
//! when something changed.
//!
//! Mouse and keys on the spectrogram:
//! * drag: mark a rectangle (time × frequency); the analysis immediately
//!   shows what it thinks belongs to it (red / orange),
//! * click: move the play head,
//! * wheel: zoom time, Shift+wheel: scroll, Ctrl+wheel: zoom frequency,
//! * Space: play / pause (the selection, if there is one), Esc: clear the
//!   selection.

// Every item, public or private, carries a doc comment (project rule).
#![warn(missing_docs, clippy::missing_docs_in_private_items)]

mod player;
mod view;

use noiseextractor_core::analysis::{analyze_selection, RegionAnalysis, Selection};
use noiseextractor_core::audio::WavStreamWriter;
use noiseextractor_core::engine::process_all;
use noiseextractor_core::filter::{Filter, FilterAction, FilterKind, Project};
use noiseextractor_core::noise::{mean_power, NoiseEstimate};
use noiseextractor_core::render::render_rgb;
use noiseextractor_core::stft::Stft;
use noiseextractor_core::{AudioData, Engine, EngineConfig, Focus};
use player::Player;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
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

/// Supported input formats for the open dialog.
const AUDIO_WILDCARD: &str = "Audio files|*.wav;*.flac;*.ogg;*.mp3;*.m4a;*.aac;*.WAV;*.FLAC;*.OGG;*.MP3|All files|*";
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
}

/// Messages from background threads to the GUI thread.
enum Background {
    /// A file finished loading (or failed).
    Loaded(PathBuf, anyhow::Result<AudioData>),
    /// The analysis of selection number `.0` finished.
    Analysed(u64, RegionAnalysis),
    /// Export progress in percent, or the final result.
    ExportProgress(usize),
    /// Export finished.
    Exported(anyhow::Result<PathBuf>),
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
    /// Counter identifying the latest selection, so stale analyses that
    /// finish late are ignored.
    selection_serial: u64,
    /// Sender handed to background threads.
    tx: Sender<Background>,
    /// Receiver polled by the timer.
    rx: Receiver<Background>,
    /// Index of the filter shown in the detail controls.
    current_filter: Option<usize>,
    /// Engine revision at the last repaint.
    last_revision: u64,
    /// Play head (samples) when not playing.
    playhead: usize,
    /// Number used for default filter names.
    filter_counter: usize,
    /// Status shown while something runs in the background.
    busy: Option<String>,
}

/// Handles to the widgets. wxDragon widgets are cheap `Copy` handles.
#[derive(Clone, Copy)]
struct Ui {
    /// Main window.
    frame: Frame,
    /// Spectrogram drawing area.
    canvas: Panel,
    /// The filter list; checked = enabled.
    filters: CheckListBox,
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
            selection_serial: 0,
            tx,
            rx,
            current_filter: Some(0),
            last_revision: 0,
            playhead: 0,
            filter_counter: 1,
            busy: None,
        }));
        let ui = build_ui();
        bind_events(&state, ui);
        refresh_filter_list(&state.borrow(), ui);
        if let Some(path) = std::env::args().nth(1) {
            start_loading(&mut state.borrow_mut(), PathBuf::from(path));
        }
        ui.frame.show(true);
        ui.frame.centre();
    });
}

/// Creates the frame, menus, spectrogram canvas and filter side panel.
fn build_ui() -> Ui {
    let frame = Frame::builder()
        .with_title("noiseextractor")
        .with_size(Size::new(1280, 760))
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
    let view_menu = Menu::builder()
        .append_item(
            ID_PLAY,
            "&Play / pause\tCtrl+P",
            "Play the selection or from the play head",
        )
        .append_item(ID_ZOOM_FIT, "Show &whole recording\tCtrl+0", "Zoom out completely")
        .append_item(
            ID_LOG_FREQ,
            "&Logarithmic frequency axis\tCtrl+L",
            "Toggle linear / logarithmic frequency axis",
        )
        .build();
    frame.set_menu_bar(
        MenuBar::builder()
            .append(file_menu, "&File")
            .append(view_menu, "&View")
            .build(),
    );
    StatusBar::builder(&frame)
        .with_fields_count(3)
        .with_status_widths(vec![-1, 260, 220])
        .add_initial_text(0, "Open an audio file to start (Ctrl+O)")
        .build();

    let root = Panel::builder(&frame).build();
    let canvas = Panel::builder(&root).build();
    canvas.set_background_style(BackgroundStyle::Paint);
    canvas.set_min_size(Size::new(400, 200));

    let side = Panel::builder(&root).build();
    let filters = CheckListBox::builder(&side).with_size(Size::new(280, 160)).build();
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
    let parts = CheckListBox::builder(&side).with_size(Size::new(280, 80)).build();
    let reduction_label = StaticText::builder(&side).with_label("Max. reduction: 18 dB").build();
    let reduction = Slider::builder(&side)
        .with_value(18)
        .with_min_value(0)
        .with_max_value(48)
        .build();
    let learn = Button::builder(&side).with_label("Learn noise from selection").build();
    let adaptive = Button::builder(&side).with_label("Adaptive noise estimate").build();
    let legend = StaticText::builder(&side)
        .with_label("Blue: remaining signal\nGreen: your selection\nRed: found to belong to it\nOrange: both\n\nDrag to select, click to set play head,\nwheel zooms (Shift scrolls, Ctrl: frequency),\nSpace plays.")
        .build();

    let s = BoxSizer::builder(Orientation::Vertical).build();
    let label = |text: &str| StaticText::builder(&side).with_label(text).build();
    s.add(&label("Filters (checked = active)"), 0, SizerFlag::All, 4);
    s.add(&filters, 1, SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right, 4);
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
    s.add(&legend, 0, SizerFlag::All, 6);
    side.set_sizer(s, true);

    let main = BoxSizer::builder(Orientation::Horizontal).build();
    main.add(&canvas, 1, SizerFlag::Expand, 0);
    main.add(&side, 0, SizerFlag::Expand, 0);
    root.set_sizer(main, true);

    Ui {
        frame,
        canvas,
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
    }
}

/// Connects all widget events to the state.
fn bind_events(state: &Shared, ui: Ui) {
    // --- menus ---
    let st = state.clone();
    ui.frame.on_menu(move |e| match e.get_id() {
        ID_OPEN_AUDIO => {
            if let Some(path) = file_dialog(ui, "Open audio", AUDIO_WILDCARD, false) {
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
                        ui.frame.set_status_text(&format!("Saved {}", path.display()), 0);
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
        ID_QUIT => ui.frame.close(true),
        _ => e.skip(true),
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
                request_repaint();
            }
            Some(WXK_HOME) => s.playhead = 0,
            _ => e.skip(true),
        }
    });

    // --- filter list and details ---
    let st = state.clone();
    ui.filters.on_selected(move |e| {
        if let Some(i) = e.get_selection() {
            let mut s = st.borrow_mut();
            s.current_filter = Some(i as usize);
            refresh_details(&s, ui);
        }
    });
    let st = state.clone();
    ui.filters.on_toggled(move |e| {
        let Some(i) = e.get_selection() else { return };
        let mut s = st.borrow_mut();
        if let Some(f) = s.project.filters.get_mut(i as usize) {
            f.enabled = ui.filters.is_checked(i);
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
        s.current_filter = if s.project.filters.is_empty() {
            None
        } else {
            Some(i.min(s.project.filters.len() - 1))
        };
        push_project(&s);
        refresh_filter_list(&s, ui);
    });
    let st = state.clone();
    ui.up.on_click(move |_| move_filter(&st, ui, -1));
    let st = state.clone();
    ui.down.on_click(move |_| move_filter(&st, ui, 1));
    let st = state.clone();
    ui.learn.on_click(move |_| learn_noise(&st, ui));
    let st = state.clone();
    ui.adaptive.on_click(move |_| {
        with_current_filter(&st, ui, |f| {
            if let FilterKind::NoiseReduction(nr) = &mut f.kind {
                nr.estimate = NoiseEstimate::default();
            }
        });
        ui.frame.set_status_text("Noise estimate: adaptive", 0);
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
        let result = AudioData::load(&path);
        let _ = tx.send(Background::Loaded(path, result));
    });
}

/// Installs a freshly loaded recording: new engine, new player, fitted view.
fn install_audio(s: &mut State, ui: Ui, path: &Path, audio: AudioData) {
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
            ui.frame.set_status_text(&format!("Playback unavailable: {e:#}"), 0);
            None
        }
    };
    let name = path
        .file_name()
        .map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
    ui.frame.set_title(&format!("{name} - noiseextractor"));
    let (w, _) = canvas_size(ui);
    s.viewport = Some(Viewport::fit(audio.len(), audio.sample_rate, w));
    s.selection = None;
    s.playhead = 0;
    s.doc = Some(Doc {
        audio,
        engine,
        player,
        name,
    });
    s.last_revision = u64::MAX;
}

/// Loads a project file, keeping the current recording.
fn open_project(st: &Shared, ui: Ui, path: &Path) {
    let mut project = match Project::load(path) {
        Ok(p) => p,
        Err(e) => return error_box(ui, &format!("{e:#}")),
    };
    let mut s = st.borrow_mut();
    s.project_path = Some(path.to_path_buf());
    s.current_filter = if project.filters.is_empty() { None } else { Some(0) };
    if let Some(doc) = &s.doc {
        project.prepare(&doc.audio);
        if project.stft != doc.engine.project().stft {
            // A different STFT needs a new engine; reinstalling is simplest.
            let audio = (*doc.audio).clone();
            let name = doc.name.clone();
            s.project = project;
            install_audio(&mut s, ui, Path::new(&name), audio);
        } else {
            s.project = project;
            push_project(&s);
        }
    } else {
        s.project = project;
    }
    refresh_filter_list(&s, ui);
}

/// Hands the current project to the engine, which recomputes in the
/// background (nearest to the focus first).
fn push_project(s: &State) {
    if let Some(doc) = &s.doc {
        doc.engine.set_project(s.project.clone());
    }
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
    s.selection_serial += 1;
    // Show the rectangle immediately; the analysis fills in the masks.
    s.project.preview = Some(Filter {
        enabled: false,
        ..Filter::event("Selection", sel, FilterAction::Suppress)
    });
    let Some(doc) = &s.doc else { return };
    let mono = doc.engine.mono().clone();
    let sr = doc.audio.sample_rate;
    let cfg = s.project.stft;
    let noise = s.project.analysis_noise();
    let params = Default::default();
    let serial = s.selection_serial;
    let tx = s.tx.clone();
    std::thread::spawn(move || {
        let a = analyze_selection(&mono, sr, &Stft::new(cfg), &sel, &noise, &params);
        let _ = tx.send(Background::Analysed(serial, a));
    });
}

/// Turns the analysed selection into a new filter.
fn add_filter_from_selection(st: &Shared, ui: Ui, action: FilterAction) {
    let mut s = st.borrow_mut();
    let Some(preview) = s.project.preview.take() else {
        ui.frame.set_status_text("Drag a rectangle over the noise first", 0);
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
    s.current_filter = Some(s.project.filters.len() - 1);
    s.selection = None;
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
    s.current_filter = Some(j as usize);
    push_project(&s);
    refresh_filter_list(&s, ui);
}

/// Sets the noise profile of the current (or first) noise-reduction filter
/// to the mean spectrum of the selected time range.
fn learn_noise(st: &Shared, ui: Ui) {
    let mut s = st.borrow_mut();
    let (Some(sel), Some(doc)) = (s.selection, s.doc.as_ref()) else {
        ui.frame
            .set_status_text("Select a stretch that contains only background noise first", 0);
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
    s.current_filter = Some(i);
    push_project(&s);
    refresh_filter_list(&s, ui);
    ui.frame.set_status_text("Noise profile learned from the selection", 0);
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
            Background::Loaded(path, Ok(audio)) => {
                s.busy = None;
                install_audio(s, ui, &path, audio);
            }
            Background::Loaded(path, Err(e)) => {
                s.busy = None;
                ui.frame
                    .set_status_text(&format!("Cannot open {}: {e:#}", path.display()), 0);
            }
            Background::Analysed(serial, a) => {
                if serial == s.selection_serial {
                    if let Some(Filter {
                        kind: FilterKind::Event(e),
                        ..
                    }) = s.project.preview.as_mut()
                    {
                        let found: Vec<String> = a.partials.iter().map(|p| format!("{:.2}", p.ratio)).collect();
                        ui.frame.set_status_text(
                            &if found.is_empty() {
                                "No related partials found".to_string()
                            } else {
                                format!("Related partials at {} x the tracked pitch", found.join(", "))
                            },
                            0,
                        );
                        e.analysis = Some(Arc::new(a));
                        push_project(s);
                    }
                }
            }
            Background::ExportProgress(p) => s.busy = Some(format!("Exporting ... {p}%")),
            Background::Exported(r) => {
                s.busy = None;
                ui.frame.set_status_text(
                    &match r {
                        Ok(p) => format!("Exported {}", p.display()),
                        Err(e) => format!("Export failed: {e:#}"),
                    },
                    0,
                );
            }
        }
    }

    let (w, _) = canvas_size(ui);
    let Some(doc) = &s.doc else {
        if let Some(b) = &s.busy {
            ui.frame.set_status_text(b, 0);
        }
        return;
    };
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
    ui.filters.clear();
    for (i, f) in s.project.filters.iter().enumerate() {
        let kind = match (&f.kind, f.action) {
            (FilterKind::NoiseReduction(_), FilterAction::Suppress) => "denoise",
            (FilterKind::NoiseReduction(_), FilterAction::Extract) => "keep noise",
            (FilterKind::Event(_), FilterAction::Suppress) => "suppress",
            (FilterKind::Event(_), FilterAction::Extract) => "extract",
        };
        ui.filters.append(&format!("{} [{kind}]", f.name));
        ui.filters.check(i as u32, f.enabled);
    }
    if let Some(i) = s.current_filter.filter(|&i| i < s.project.filters.len()) {
        ui.filters.set_selection(i as u32, true);
    }
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
    let Some(f) = f else {
        ui.name.change_value("");
        ui.reduction.enable(false);
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
        ui.parts.append(label);
        ui.parts.check(i as u32, on);
    }
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
