//! File > Recent recordings and File > Recent projects: lists of recently
//! opened files, remembered between runs.
//!
//! wxWidgets has `wxFileHistory` for this, but wxdragon does not bind it,
//! so the list is kept here ([`RecentFiles`], pure and unit-tested). It is
//! stored in the per-user config store as `/RecentAudio/File1` … `File9`
//! and `/RecentProjects/File1` …, the layout `wxFileHistory` would use.
//! This follows losslessaudiocut's File > Open recent on purpose; keep
//! them alike.
//!
//! The newest file comes first; opening a file again moves it to the
//! front instead of listing it twice.

use std::path::{Path, PathBuf};

use wxdragon::prelude::*;

use crate::panes::config;

/// How many files a list holds, like `wxFileHistory`'s default. Nine also
/// means every entry gets a single-digit mnemonic (`&1` … `&9`).
pub const MAX_FILES: usize = 9;

/// Which of the two lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Recordings (File > Open audio).
    Audio,
    /// Filter projects (File > Open project / Save project).
    Project,
}

impl Kind {
    /// Both lists, in menu order.
    pub const ALL: [Kind; 2] = [Kind::Audio, Kind::Project];

    /// Config group the list is stored in.
    fn group(self) -> &'static str {
        match self {
            Kind::Audio => "/RecentAudio",
            Kind::Project => "/RecentProjects",
        }
    }

    /// Menu id of "Clear list" in this kind's submenu; the file entries
    /// follow it, newest first (see [`Kind::entry_id`]).
    ///
    /// The ids sit between the fixed commands in `main.rs` (up to about
    /// `ID_HIGHEST + 30`) and the panes' (`ID_HIGHEST + 100`), with room
    /// for both lists.
    pub fn clear_id(self) -> i32 {
        match self {
            Kind::Audio => ID_HIGHEST + 50,
            Kind::Project => ID_HIGHEST + 70,
        }
    }

    /// Menu id of entry `index` (0 = newest).
    fn entry_id(self, index: usize) -> i32 {
        self.clear_id() + 1 + index as i32
    }

    /// Submenu label.
    pub fn menu_label(self) -> &'static str {
        match self {
            Kind::Audio => "Recent &recordings",
            Kind::Project => "Recent pro&jects",
        }
    }

    /// Submenu help text.
    pub fn menu_help(self) -> &'static str {
        match self {
            Kind::Audio => "Open one of the recently opened recordings",
            Kind::Project => "Open one of the recently used filter projects",
        }
    }

    /// Config key of entry `index` (0 = newest).
    fn key(self, index: usize) -> String {
        format!("{}/File{}", self.group(), index + 1)
    }
}

/// What a menu id of one of the submenus means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// Open entry `.1` (0 = newest) of list `.0`.
    Open(Kind, usize),
    /// Empty list `.0`.
    Clear(Kind),
}

impl Command {
    /// The command behind menu id `id`, if it belongs to a recent-files
    /// submenu.
    pub fn for_menu_id(id: i32) -> Option<Command> {
        Kind::ALL.into_iter().find_map(|k| {
            if id == k.clear_id() {
                return Some(Command::Clear(k));
            }
            let i = usize::try_from(id - k.entry_id(0)).ok()?;
            (i < MAX_FILES).then_some(Command::Open(k, i))
        })
    }
}

/// Recently opened files, newest first, at most [`MAX_FILES`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecentFiles {
    /// Newest first, no duplicates.
    paths: Vec<PathBuf>,
}

/// One item of a recent-files submenu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuEntry {
    /// The label, with a mnemonic and `&` escaped as `&&`.
    pub label: String,
    /// The full path, for the status bar while the item is highlighted.
    pub help: String,
}

impl RecentFiles {
    /// A list from stored paths, newest first. Duplicates and paths
    /// beyond [`MAX_FILES`] are dropped, so a hand-edited or corrupt store
    /// still gives a valid list.
    pub fn from_paths(paths: impl IntoIterator<Item = PathBuf>) -> Self {
        let mut list = RecentFiles::default();
        // Adding moves to the front, so add the oldest first.
        let paths: Vec<PathBuf> = paths.into_iter().collect();
        for p in paths.into_iter().rev() {
            list.add(p);
        }
        list
    }

    /// The files, newest first.
    #[cfg(test)]
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Whether no file is listed.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// The file of entry `index` (0 = newest), if there is one.
    pub fn get(&self, index: usize) -> Option<&Path> {
        self.paths.get(index).map(PathBuf::as_path)
    }

    /// Puts `path` at the front: either new, or moved up from further
    /// down. The oldest file drops off when the list is full. Empty paths
    /// are ignored. A relative path is made absolute first, so the same
    /// file opened from different working directories (e.g. from the
    /// command line) is listed once.
    pub fn add(&mut self, path: PathBuf) {
        if path.as_os_str().is_empty() {
            return;
        }
        let path = std::path::absolute(&path).unwrap_or(path);
        self.paths.retain(|p| *p != path);
        self.paths.insert(0, path);
        self.paths.truncate(MAX_FILES);
    }

    /// Takes `path` off the list (e.g. because it no longer exists).
    /// Returns whether it was listed.
    pub fn remove(&mut self, path: &Path) -> bool {
        let before = self.paths.len();
        self.paths.retain(|p| p != path);
        self.paths.len() != before
    }

    /// Empties the list.
    pub fn clear(&mut self) {
        self.paths.clear();
    }

    /// The submenu's items, newest first.
    ///
    /// A label is `&N name`: the file name alone keeps the menu narrow.
    /// Where two listed files have the same name (the same recording in
    /// different folders), both show their full path instead, so they can
    /// be told apart.
    pub fn menu_entries(&self) -> Vec<MenuEntry> {
        let name = |p: &Path| {
            p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.to_string_lossy().into_owned())
        };
        self.paths
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let short = name(p);
                let ambiguous = self.paths.iter().filter(|q| *q != p).any(|q| name(q) == short);
                let shown = if ambiguous {
                    p.to_string_lossy().into_owned()
                } else {
                    short
                };
                MenuEntry {
                    label: format!("&{} {}", i + 1, escape_mnemonics(&shown)),
                    help: p.to_string_lossy().into_owned(),
                }
            })
            .collect()
    }
}

/// Escapes `&` (which marks a mnemonic in wxWidgets menu labels) and
/// replaces tabs (which start the accelerator part of a label).
fn escape_mnemonics(s: &str) -> String {
    s.replace('&', "&&").replace('\t', " ")
}

/// The stored list of `kind` (empty if there is none).
pub fn load(kind: Kind) -> RecentFiles {
    let c = config();
    let paths = (0..MAX_FILES)
        .map(|i| c.read_string(&kind.key(i), ""))
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    RecentFiles::from_paths(paths)
}

/// Stores the list of `kind`. Failing to store it only means it is not
/// remembered, so errors are ignored.
pub fn save(kind: Kind, list: &RecentFiles) {
    let c = config();
    for i in 0..MAX_FILES {
        match list.get(i) {
            Some(p) => {
                c.write_string(&kind.key(i), &p.to_string_lossy());
            }
            None => {
                c.delete_entry(&kind.key(i), true);
            }
        }
    }
    c.flush(false);
}

/// The empty submenu of `kind`; [`refresh`] fills it.
pub fn submenu(kind: Kind) -> Menu {
    Menu::builder()
        .append_separator()
        .append_item(kind.clear_id(), "&Clear list", "Forget the files in this list")
        .build()
}

/// Rebuilds the file entries of `kind`'s submenu in `bar` from `list`.
pub fn refresh(bar: &MenuBar, kind: Kind, list: &RecentFiles) {
    // The submenu is the menu holding "Clear list"; no need to keep a
    // handle of it.
    let Some((_, menu)) = bar.find_item_and_menu(kind.clear_id()) else {
        return;
    };
    for i in 0..MAX_FILES {
        let id = kind.entry_id(i);
        if menu.find_item(id).is_some() {
            menu.delete(id);
        }
    }
    for (i, e) in list.menu_entries().iter().enumerate() {
        menu.insert(i, kind.entry_id(i), &e.label, &e.help, ItemKind::Normal);
    }
    menu.enable_item(kind.clear_id(), !list.is_empty());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path that is absolute on every platform (Windows needs a drive),
    /// in the form [`RecentFiles::add`] stores it: on Windows,
    /// `std::path::absolute` also turns `/` into `\\`.
    fn p(s: &str) -> PathBuf {
        let raw = if cfg!(windows) { format!("C:{s}") } else { s.to_string() };
        std::path::absolute(&raw).unwrap_or_else(|_| PathBuf::from(raw))
    }

    #[test]
    fn newest_first_without_duplicates() {
        let mut r = RecentFiles::default();
        r.add(p("/a/one.flac"));
        r.add(p("/a/two.wav"));
        r.add(p("/a/one.flac"));
        assert_eq!(r.paths(), [p("/a/one.flac"), p("/a/two.wav")]);
        assert_eq!(r.get(1), Some(p("/a/two.wav").as_path()));
        assert_eq!(r.get(2), None);
    }

    #[test]
    fn oldest_drops_off_when_full() {
        let mut r = RecentFiles::default();
        for i in 0..=MAX_FILES {
            r.add(p(&format!("/f{i}.wav")));
        }
        assert_eq!(r.paths().len(), MAX_FILES);
        assert_eq!(r.get(0), Some(p("/f9.wav").as_path()));
        assert!(!r.paths().contains(&p("/f0.wav")));
    }

    #[test]
    fn empty_paths_are_ignored_and_relative_ones_made_absolute() {
        let mut r = RecentFiles::default();
        r.add(PathBuf::new());
        assert!(r.is_empty());
        r.add(PathBuf::from("owl.flac"));
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(r.paths(), [cwd.join("owl.flac")]);
        // The same file given as an absolute path is not listed twice.
        r.add(cwd.join("owl.flac"));
        assert_eq!(r.paths().len(), 1);
    }

    #[test]
    fn remove_and_clear() {
        let mut r = RecentFiles::from_paths([p("/x.ogg"), p("/y.ogg")]);
        assert!(r.remove(&p("/x.ogg")));
        assert!(!r.remove(&p("/x.ogg")));
        assert_eq!(r.paths(), [p("/y.ogg")]);
        r.clear();
        assert!(r.is_empty());
    }

    #[test]
    fn from_paths_keeps_order_and_repairs_bad_stores() {
        let stored: Vec<PathBuf> = ["/new", "/mid", "/new", "/old", "/1", "/2", "/3", "/4", "/5", "/6", "/7"]
            .iter()
            .map(|s| p(s))
            .collect();
        let r = RecentFiles::from_paths(stored);
        assert_eq!(r.paths().len(), MAX_FILES);
        // The first occurrence of a duplicate wins: it was the newer one.
        assert_eq!(&r.paths()[..3], [p("/new"), p("/mid"), p("/old")]);
        assert_eq!(r.get(8), Some(p("/6").as_path()));
    }

    #[test]
    fn labels_have_mnemonics_and_escape_ampersands() {
        let r = RecentFiles::from_paths([p("/rec/Owl & Fox.flac"), p("/rec/rain.wav")]);
        let e = r.menu_entries();
        assert_eq!(e[0].label, "&1 Owl && Fox.flac");
        assert_eq!(e[0].help, p("/rec/Owl & Fox.flac").to_string_lossy());
        assert_eq!(e[1].label, "&2 rain.wav");
    }

    #[test]
    fn same_names_show_full_paths() {
        let r = RecentFiles::from_paths([p("/mon/night.flac"), p("/tue/night.flac"), p("/tue/other.flac")]);
        let labels: Vec<String> = r.menu_entries().into_iter().map(|e| e.label).collect();
        assert_eq!(
            labels,
            [
                format!("&1 {}", p("/mon/night.flac").display()),
                format!("&2 {}", p("/tue/night.flac").display()),
                "&3 other.flac".to_string(),
            ]
        );
    }

    #[test]
    fn menu_ids_map_back_and_stay_clear_of_other_commands() {
        for k in Kind::ALL {
            assert_eq!(Command::for_menu_id(k.clear_id()), Some(Command::Clear(k)));
            for i in 0..MAX_FILES {
                assert_eq!(Command::for_menu_id(k.entry_id(i)), Some(Command::Open(k, i)));
            }
            assert_eq!(Command::for_menu_id(k.entry_id(MAX_FILES)), None);
        }
        // The two lists' ids do not overlap ...
        assert!(Kind::Audio.entry_id(MAX_FILES - 1) < Kind::Project.clear_id());
        // ... nor reach the panes' (ID_HIGHEST + 100) ...
        assert!(Kind::Project.entry_id(MAX_FILES - 1) < ID_HIGHEST + 100);
        // ... nor the fixed commands.
        assert_eq!(Command::for_menu_id(ID_HIGHEST + 1), None);
    }

    #[test]
    fn config_keys_are_numbered_from_one() {
        assert_eq!(Kind::Audio.key(0), "/RecentAudio/File1");
        assert_eq!(Kind::Project.key(8), "/RecentProjects/File9");
    }
}
