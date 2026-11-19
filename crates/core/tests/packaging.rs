//! The packages' file associations must match
//! [`noiseextractor_core::audio::FILE_TYPES`].
//!
//! The Linux desktop file, the macOS `Info.plist` and the Windows installer
//! script each list the supported types in their own syntax. These tests
//! read them from the repository and compare, so adding a format to
//! `FILE_TYPES` fails here until every package offers it, and a package
//! cannot claim a type the program does not open. Projects
//! ([`PROJECT_FILE_TYPE`]) are registered as a type of our own everywhere:
//! a MIME type declared in `packaging/linux/noiseextractor.xml`, a UTI the
//! macOS app exports, and a Windows ProgID that owns the extension. The
//! snap's own desktop file must equal the Linux one apart from its icon. (Adapted from
//! losslessaudiocut's `crates/lac-app/tests/packaging.rs`.)

use noiseextractor_core::audio::FILE_TYPES;
use noiseextractor_core::project_file::PROJECT_FILE_TYPE;
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
        .chain([&PROJECT_FILE_TYPE])
        .flat_map(|t| t.mime_types)
        .map(|m| format!("{m};"))
        .collect();
    assert_eq!(desktop_value(&f, "MimeType"), Some(expected.as_str()));
    // %F: the file manager passes every selected file, so a recording and
    // a project can be opened together (project_file::StartFiles sorts them).
    let exec = desktop_value(&f, "Exec").expect("Exec=");
    assert_eq!(exec, "noiseextractor %F");
}

/// The value of attribute `attr` in every element `<tag ...>` of `xml`.
fn xml_attributes<'a>(xml: &'a str, tag: &str, attr: &str) -> Vec<&'a str> {
    xml.split(&format!("<{tag} "))
        .skip(1)
        .map(|rest| {
            let start = rest.find(&format!("{attr}=\"")).expect("attribute") + attr.len() + 2;
            let rest = &rest[start..];
            &rest[..rest.find('"').expect("closing quote")]
        })
        .collect()
}

#[test]
fn shared_mime_info_declares_the_project_type() {
    let xml = repo_file("packaging/linux/noiseextractor.xml");
    assert_eq!(xml_attributes(&xml, "mime-type", "type"), PROJECT_FILE_TYPE.mime_types);
    let globs: Vec<String> = PROJECT_FILE_TYPE.extensions.iter().map(|e| format!("*.{e}")).collect();
    assert_eq!(xml_attributes(&xml, "glob", "pattern"), globs);
    assert_eq!(xml_attributes(&xml, "sub-class-of", "type"), ["application/json"]);
    // The icon theme name is the MIME type with "/" replaced by "-"; the
    // packages install the icon under that name (deb.sh, appimage.sh).
    let icon = PROJECT_FILE_TYPE.mime_types[0].replace('/', "-");
    assert_eq!(xml_attributes(&xml, "icon", "name"), [icon.as_str()]);
    for script in ["packaging/linux/deb.sh", "packaging/linux/appimage.sh"] {
        let text = repo_file(script);
        assert!(
            text.contains(&format!("mimetypes/{icon}")),
            "{script} installs the icon"
        );
        assert!(text.contains("noiseextractor.xml"), "{script} ships the MIME type");
    }
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
    // The first document type is audio, offered but never the default
    // handler; the second is the project, which the app owns.
    let doc_types: Vec<&str> = plist.split("<key>LSHandlerRank</key>").skip(1).collect();
    assert_eq!(doc_types.len(), 2);
    assert!(doc_types[0].trim_start().starts_with("<string>Alternate</string>"));
    assert_eq!(plist_strings_after(doc_types[0], "<key>LSItemContentTypes</key>"), utis);
    assert!(doc_types[1].trim_start().starts_with("<string>Owner</string>"));
    assert_eq!(
        plist_strings_after(doc_types[1], "<key>LSItemContentTypes</key>"),
        [PROJECT_FILE_TYPE.uti]
    );

    // The project type is exported with its extension and MIME type.
    let exported = &plist[plist
        .find("<key>UTExportedTypeDeclarations</key>")
        .expect("exported types")..];
    let exported = &exported[..exported.find("<key>UTImportedTypeDeclarations</key>").expect("order")];
    assert_eq!(exported.matches("<key>UTTypeIdentifier</key>").count(), 1);
    assert_eq!(
        plist_strings_after(exported, "<key>UTTypeIdentifier</key>")[0],
        PROJECT_FILE_TYPE.uti
    );
    assert_eq!(
        plist_strings_after(exported, "<key>public.filename-extension</key>"),
        PROJECT_FILE_TYPE.extensions
    );
    assert_eq!(
        plist_strings_after(exported, "<key>public.mime-type</key>"),
        PROJECT_FILE_TYPE.mime_types
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
    let mut all = extensions();
    all.extend(PROJECT_FILE_TYPE.extensions.iter().map(|e| e.to_string()));
    assert_eq!(iss_extensions(&iss, "\\OpenWithProgids\"", "Software\\Classes\\."), all);
    assert_eq!(iss_extensions(&iss, "\\SupportedTypes\"", "ValueName: \"."), all);
    // Audio files only get "Open with" entries: no audio extension's
    // default program changes. Only the project extension, which is ours,
    // gets noiseextractor as its default program.
    let project_key = format!("Software\\Classes\\.{}\"", PROJECT_FILE_TYPE.extensions[0]);
    for l in iss.lines().filter(|l| l.contains("Software\\Classes\\.")) {
        assert!(l.contains("\\OpenWithProgids\"") || l.contains(&project_key), "{l}");
    }
    assert!(iss.contains(&format!(
        "Subkey: \"{project_key}; ValueType: string; ValueName: \"\"; ValueData: \"noiseextractor.project\""
    )));
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
