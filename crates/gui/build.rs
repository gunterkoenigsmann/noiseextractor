//! Build script of the GUI: on Windows, compiles the resources of
//! `noiseextractor.exe` (icon, version information, manifest; see
//! `build/resources.rs`) and links them into the program. Does nothing on
//! other platforms.

#[path = "build/resources.rs"]
mod resources;

use std::path::PathBuf;

fn main() {
    // `cfg!(windows)` would describe the build machine; the target is what
    // counts when cross-compiling.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let manifest_dir = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let packaging = manifest_dir.join("../../packaging");
    let icon = packaging.join("icons/noiseextractor.ico");
    let manifest = packaging.join("windows/noiseextractor.exe.manifest");
    for file in [&icon, &manifest] {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    println!("cargo:rerun-if-changed=build/resources.rs");

    let script = resources::rc_script(&env("CARGO_PKG_VERSION"), &icon, &manifest)
        .unwrap_or_else(|e| panic!("cannot build the Windows resources: {e}"));
    let rc = PathBuf::from(env("OUT_DIR")).join("noiseextractor.rc");
    std::fs::write(&rc, script).unwrap_or_else(|e| panic!("cannot write {}: {e}", rc.display()));
    // Only the GUI binary gets them, not the test harness.
    embed_resource::compile_for(&rc, ["noiseextractor"], embed_resource::NONE)
        .manifest_required()
        .unwrap_or_else(|e| panic!("cannot compile {}: {e}", rc.display()));
}

/// A variable Cargo always sets for build scripts.
fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("cargo did not set {name}"))
}
