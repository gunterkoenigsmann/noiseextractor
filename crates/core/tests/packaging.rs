//! The packages' file associations must match
//! [`noiseextractor_core::audio::FILE_TYPES`].
//!
//! The Linux desktop file, the macOS `Info.plist` and the Windows installer
//! script each list the supported types in their own syntax. These tests
//! read them from the repository and compare, so adding a format to
//! `FILE_TYPES` fails here until every package offers it, and a package
//! cannot claim a type the program does not open. The snap's own desktop
//! file must equal the Linux one apart from its icon. (Adapted from
//! losslessaudiocut's `crates/lac-app/tests/packaging.rs`.)

use noiseextractor_core::audio::FILE_TYPES;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Reads a file relative to the repository root.
fn repo_file(path: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(path);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The value of `key=` in a desktop file.
fn desktop_value<'a>(file: &'a str, key: &str) -> Option<&'a str> {
    file.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix('='))
}

/// Every extension of [`FILE_TYPES`].
fn extensions() -> BTreeSet<String> {
    FILE_TYPES
        .iter()
        .flat_map(|t| t.extensions)
        .map(|e| e.to_string())
        .collect()
}

#[test]
fn desktop_file_offers_exactly_the_supported_mime_types() {
    let f = repo_file("packaging/linux/noiseextractor.desktop");
    let expected: String = FILE_TYPES
        .iter()
        .flat_map(|t| t.mime_types)
        .map(|m| format!("{m};"))
        .collect();
    assert_eq!(desktop_value(&f, "MimeType"), Some(expected.as_str()));
    // %f: the file manager passes one file, which the GUI opens (it reads
    // the first argument).
    let exec = desktop_value(&f, "Exec").expect("Exec=");
    assert_eq!(exec, "noiseextractor %f");
}

/// The `<string>` values between `start` and the next `</array>`.
fn plist_strings_after<'a>(text: &'a str, start: &str) -> Vec<&'a str> {
    let Some(i) = text.find(start) else {
        return Vec::new();
    };
    let rest = &text[i + start.len()..];
    let block = &rest[..rest.find("</array>").expect("</array>")];
    block
        .split("<string>")
        .skip(1)
        .map(|s| s.split("</string>").next().expect("</string>"))
        .collect()
}

/// UTIs that macOS defines itself, so `Info.plist` does not import them.
const SYSTEM_UTIS: &[&str] = &[
    "public.mp3",
    "com.microsoft.waveform-audio",
    "public.mpeg-4-audio",
    "public.aac-audio",
];

#[test]
fn macos_document_types_match() {
    let plist = repo_file("packaging/macos/Info.plist");
    let utis: Vec<&str> = FILE_TYPES.iter().map(|t| t.uti).collect();
    assert_eq!(plist_strings_after(&plist, "<key>LSItemContentTypes</key>"), utis);
    assert!(
        plist.contains("<string>Alternate</string>"),
        "never the default handler"
    );

    // Every type macOS may not know is declared with its extensions and
    // MIME types; nothing else is declared.
    let imports = &plist[plist
        .find("<key>UTImportedTypeDeclarations</key>")
        .expect("imported types")..];
    let declared: Vec<&str> = imports
        .split("<key>UTTypeIdentifier</key>")
        .skip(1)
        .map(|decl| {
            let uti = plist_strings_after(decl, "")[0];
            let t = FILE_TYPES
                .iter()
                .find(|t| t.uti == uti)
                .unwrap_or_else(|| panic!("{uti} is not a supported type"));
            assert_eq!(
                plist_strings_after(decl, "<key>public.filename-extension</key>"),
                t.extensions,
                "{uti}"
            );
            assert_eq!(
                plist_strings_after(decl, "<key>public.mime-type</key>"),
                t.mime_types,
                "{uti}"
            );
            uti
        })
        .collect();
    let expected: Vec<&str> = utis.iter().copied().filter(|u| !SYSTEM_UTIS.contains(u)).collect();
    assert_eq!(declared, expected);
}

/// The extensions in Inno Setup registry lines that contain `marker`, taken
/// from the text between `before` and the next `\` or `"`.
fn iss_extensions(iss: &str, marker: &str, before: &str) -> BTreeSet<String> {
    iss.lines()
        .filter(|l| l.starts_with("Root:") && l.contains(marker))
        .map(|l| {
            let rest = &l[l.find(before).expect("extension") + before.len()..];
            rest[..rest.find(['\\', '"']).expect("end")].to_string()
        })
        .collect()
}

#[test]
fn windows_installer_offers_exactly_the_supported_extensions() {
    let iss = repo_file("packaging/windows/noiseextractor.iss");
    assert_eq!(
        iss_extensions(&iss, "\\OpenWithProgids\"", "Software\\Classes\\."),
        extensions()
    );
    assert_eq!(
        iss_extensions(&iss, "\\SupportedTypes\"", "ValueName: \"."),
        extensions()
    );
    // Only "Open with" entries: no extension's default program changes.
    for l in iss.lines().filter(|l| l.contains("Software\\Classes\\.")) {
        assert!(l.contains("\\OpenWithProgids\""), "{l}");
    }
}

#[test]
fn smoke_test_recording_loads() {
    // The release workflow runs the packaged CLI on this file.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packaging/smoke-test.wav");
    let audio = noiseextractor_core::AudioData::load(&path).expect("loads");
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.len(), 16_000);
}

#[test]
fn snap_desktop_file_matches_the_linux_one() {
    // The snap ships its own desktop file (snap/gui) because its icon lives
    // in the snap's meta/gui instead of the host's icon theme; everything
    // else, above all the MIME types, must stay the same as in the .deb and
    // the AppImage.
    let linux = repo_file("packaging/linux/noiseextractor.desktop");
    let snap = repo_file("snap/gui/noiseextractor.desktop");
    for key in [
        "Type",
        "Name",
        "GenericName",
        "Comment",
        "Exec",
        "Terminal",
        "Categories",
        "Keywords",
        "MimeType",
    ] {
        assert_eq!(desktop_value(&snap, key), desktop_value(&linux, key), "{key}");
    }
    assert_eq!(
        desktop_value(&snap, "Icon"),
        Some("${SNAP}/meta/gui/noiseextractor.svg")
    );
}

#[test]
fn snap_icon_is_the_app_icon() {
    // snapcraft copies snap/gui into meta/gui; a symlink to the icon in
    // packaging/icons would not survive a checkout on Windows, so it is a
    // copy, and this keeps the copy current.
    assert_eq!(
        repo_file("snap/gui/noiseextractor.svg"),
        repo_file("packaging/icons/noiseextractor.svg")
    );
}
