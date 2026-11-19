//! Project files on disk: their own extension, how the desktops know them,
//! and how the GUI tells them from recordings on its command line.
//!
//! A project is saved as JSON (see [`Project::save`](crate::Project::save))
//! under the extension [`PROJECT_EXTENSION`]. The extension exists so that
//! file managers, Finder and Explorer can open projects with the program and
//! shells can complete them; the content is still the same JSON as before,
//! and older projects saved as `*.json` or `*.nxp.json` keep loading.

use crate::audio::FileType;
use std::io::Read;
use std::path::{Path, PathBuf};

/// File name extension of project files, without the dot.
pub const PROJECT_EXTENSION: &str = "nxp";

/// Project files as the packages register them: the extension, a MIME type
/// of our own (declared in `packaging/linux/noiseextractor.xml`, a subclass
/// of `application/json`) and a UTI the macOS app exports.
/// `tests/packaging.rs` checks the packages against this.
pub const PROJECT_FILE_TYPE: FileType = FileType {
    extensions: &[PROJECT_EXTENSION],
    mime_types: &["application/x-noiseextractor-project"],
    uti: "io.github.gunterkoenigsmann.noiseextractor.project",
};

/// The patterns of the GUI's project dialogs: the current extension first,
/// then what earlier versions saved (`*.nxp.json`, plain `*.json`). GTK
/// matches case-sensitively, hence the upper-case variants.
pub fn project_dialog_patterns() -> String {
    let e = PROJECT_EXTENSION;
    [
        format!("*.{e}"),
        format!("*.{}", e.to_uppercase()),
        "*.json".to_string(),
        "*.JSON".to_string(),
    ]
    .join(";")
}

/// `path` with the project extension added if it has none, so that a name
/// typed into a save dialog without extension still becomes a project file
/// the desktop recognises. A name with any extension is left alone: the
/// user chose it (for example `old.json`).
pub fn with_project_extension(path: &Path) -> PathBuf {
    if path.extension().is_some() {
        path.to_path_buf()
    } else {
        path.with_extension(PROJECT_EXTENSION)
    }
}

/// Whether `path` names a project file rather than a recording.
///
/// The extension decides when it is ours or `json` (the extension of older
/// projects). Anything else is decided by the content, so that a project
/// renamed or saved without extension still opens as one: projects are JSON
/// objects and start with `{` after optional white space, which no audio
/// format the program reads does (WAV starts with `RIFF`, FLAC with `fLaC`,
/// Ogg with `OggS`, MP3 with `ID3` or a frame sync byte 0xFF, MP4 with a box
/// size, ADTS with 0xFF). An unreadable file counts as audio, so that its
/// error message comes from the audio loader.
pub fn is_project_file(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        if ext.eq_ignore_ascii_case(PROJECT_EXTENSION) || ext.eq_ignore_ascii_case("json") {
            return true;
        }
    }
    let mut head = [0u8; 64];
    let Ok(n) = std::fs::File::open(path).and_then(|mut f| f.read(&mut head)) else {
        return false;
    };
    head[..n]
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|&b| b == b'{')
}

/// The files the GUI was started with, sorted into one recording and one
/// project. File managers pass several files when several are selected
/// ("Exec=noiseextractor %F"), so a recording and its project can be opened
/// together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StartFiles {
    /// The recording to open: the first argument that is not a project.
    pub audio: Option<PathBuf>,
    /// The project to open: the first argument that is one.
    pub project: Option<PathBuf>,
    /// Further arguments, which the GUI cannot open at the same time and
    /// only reports.
    pub ignored: Vec<PathBuf>,
}

impl StartFiles {
    /// Sorts command-line arguments with [`is_project_file`].
    pub fn from_args(args: impl IntoIterator<Item = PathBuf>) -> Self {
        let mut files = Self::default();
        for path in args {
            let slot = if is_project_file(&path) {
                &mut files.project
            } else {
                &mut files.audio
            };
            if slot.is_none() {
                *slot = Some(path);
            } else {
                files.ignored.push(path);
            }
        }
        files
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Project;

    #[test]
    fn extension_decides_for_projects_old_and_new() {
        // Existence is not needed: the extension alone decides.
        for name in ["a.nxp", "b.NXP", "old.json", "older.nxp.json"] {
            assert!(is_project_file(Path::new(name)), "{name}");
        }
    }

    #[test]
    fn content_decides_without_known_extension() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("renamed");
        Project::default().save(&project).unwrap();
        assert!(is_project_file(&project));

        let wav = dir.path().join("sound.dat");
        std::fs::write(&wav, b"RIFF\0\0\0\0WAVEfmt ").unwrap();
        assert!(!is_project_file(&wav));
        // Leading white space before the JSON object.
        let spaced = dir.path().join("spaced.txt");
        std::fs::write(&spaced, b" \n\t{\"version\": 1}").unwrap();
        assert!(is_project_file(&spaced));
        // Missing files count as audio, whose loader reports them.
        assert!(!is_project_file(&dir.path().join("missing.flac")));
    }

    #[test]
    fn save_dialog_names_get_the_extension_only_when_they_have_none() {
        assert_eq!(with_project_extension(Path::new("/x/birds")), Path::new("/x/birds.nxp"));
        assert_eq!(with_project_extension(Path::new("/x/b.nxp")), Path::new("/x/b.nxp"));
        assert_eq!(with_project_extension(Path::new("/x/b.json")), Path::new("/x/b.json"));
    }

    #[test]
    fn dialog_patterns_offer_new_and_old_projects() {
        let p = project_dialog_patterns();
        assert!(p.starts_with("*.nxp;*.NXP;"), "{p}");
        assert!(p.contains("*.json"), "{p}");
    }

    #[test]
    fn start_files_sorts_arguments() {
        let args = ["rec.flac", "birds.nxp", "other.wav", "old.json"].map(PathBuf::from);
        let files = StartFiles::from_args(args);
        assert_eq!(files.audio.as_deref(), Some(Path::new("rec.flac")));
        assert_eq!(files.project.as_deref(), Some(Path::new("birds.nxp")));
        assert_eq!(files.ignored, [PathBuf::from("other.wav"), PathBuf::from("old.json")]);
        // The order does not matter.
        let files = StartFiles::from_args(["p.nxp", "r.ogg"].map(PathBuf::from));
        assert_eq!(files.audio.as_deref(), Some(Path::new("r.ogg")));
        assert_eq!(files.project.as_deref(), Some(Path::new("p.nxp")));
        assert_eq!(StartFiles::from_args([]), StartFiles::default());
    }

    #[test]
    fn project_type_is_not_an_audio_type() {
        for t in crate::audio::FILE_TYPES {
            assert!(!t.extensions.contains(&PROJECT_EXTENSION));
        }
    }
}
