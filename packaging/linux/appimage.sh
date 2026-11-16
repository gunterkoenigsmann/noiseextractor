#!/usr/bin/env bash
# Packs the release binaries into an AppImage for the architecture of the
# machine it runs on (x86_64 or aarch64):
#
#   packaging/linux/appimage.sh VERSION OUTDIR
#
# Expects `cargo build --release -p noiseextractor -p noiseextractor-cli`
# to have run. Writes OUTDIR/noiseextractor-VERSION-linux-ARCH.AppImage.
#
# The AppImage starts the GUI, or the command line tool
# `noiseextractor-cli` with `--cli` as the first argument or when started
# through a symlink named `noiseextractor-cli` (packaging/linux/AppRun).
#
# Tools: linuxdeploy (copies every library the binaries need into the
# AppDir, except the ones its excludelist says must come from the host, such
# as glibc, libGL and libasound) and its GTK plugin (GTK 3 modules, themes,
# GSettings schemas and the environment to find them). Both are downloaded
# at a pinned version and checked against committed hashes (see "Tools").
set -euo pipefail

version="$1"
outdir="$(mkdir -p "$2" && cd "$2" && pwd)"
root="$(cd "$(dirname "$0")/../.." && pwd)"
arch="$(uname -m)"            # x86_64 or aarch64, as linuxdeploy names them
target_dir="${CARGO_TARGET_DIR:-$root/target}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# --- Tools -----------------------------------------------------------------
# Pinned to a release tag and a commit and checked against the hashes below,
# because they run in the job that builds the published packages: a changed
# upstream file must fail the build, not end up in a release. To update,
# change the version or commit, download the files and put their
# `sha256sum` here.
linuxdeploy_version="1-alpha-20251107-1"
gtk_plugin_commit="7a3fbc31a9e5075073ff8790f26effbac5f84453"
case "$arch" in
x86_64) linuxdeploy_sha256="c20cd71e3a4e3b80c3483cef793cda3f4e990aca14014d23c544ca3ce1270b4d" ;;
aarch64) linuxdeploy_sha256="620095110d693282b8ebeb244a95b5e911cf8f65f76c88b4b47d16ae6346fcff" ;;
*) echo "error: no linuxdeploy for $arch" >&2; exit 1 ;;
esac
gtk_plugin_sha256="b0f4cbc684a0103a9651f0955b635eaea0096b3a66c0f5a2c2aa337960375171"

tools="$work/tools"
mkdir -p "$tools"
curl -sSfL -o "$tools/linuxdeploy" \
    "https://github.com/linuxdeploy/linuxdeploy/releases/download/$linuxdeploy_version/linuxdeploy-$arch.AppImage"
curl -sSfL -o "$tools/linuxdeploy-plugin-gtk.sh" \
    "https://raw.githubusercontent.com/linuxdeploy/linuxdeploy-plugin-gtk/$gtk_plugin_commit/linuxdeploy-plugin-gtk.sh"
sha256sum -c - <<EOF
$linuxdeploy_sha256  $tools/linuxdeploy
$gtk_plugin_sha256  $tools/linuxdeploy-plugin-gtk.sh
EOF
chmod +x "$tools"/*
# CI runners (and containers) have no FUSE; AppImages can unpack themselves
# to a temporary directory instead. This covers linuxdeploy and the
# appimagetool inside it.
export APPIMAGE_EXTRACT_AND_RUN=1

# --- AppDir ----------------------------------------------------------------
appdir="$work/AppDir"
icons="$appdir/usr/share/icons/hicolor"
mkdir -p "$icons/scalable/apps" "$icons/256x256/apps"
cp "$root/packaging/icons/noiseextractor.svg" "$icons/scalable/apps/"
# A PNG as well: some desktops and AppImage integrators only use bitmaps.
rsvg-convert -w 256 -h 256 "$root/packaging/icons/noiseextractor.svg" \
    -o "$icons/256x256/apps/noiseextractor.png"

# The binaries' rpath is set by linuxdeploy so that they find usr/lib.
# DEPLOY_GTK_VERSION: the GTK plugin cannot tell GTK 3 from GTK 4 by itself.
export DEPLOY_GTK_VERSION=3
# The AppImage's file name; linuxdeploy's appimage plugin reads these.
export LDAI_OUTPUT="$outdir/noiseextractor-$version-linux-$arch.AppImage"
export LINUXDEPLOY_OUTPUT_VERSION="$version"
# linuxdeploy finds its plugins (linuxdeploy-plugin-*) on PATH.
export PATH="$tools:$PATH"

linuxdeploy \
    --appdir "$appdir" \
    --executable "$target_dir/release/noiseextractor" \
    --executable "$target_dir/release/noiseextractor-cli" \
    --desktop-file "$root/packaging/linux/noiseextractor.desktop" \
    --icon-file "$icons/256x256/apps/noiseextractor.png" \
    --custom-apprun "$root/packaging/linux/AppRun" \
    --plugin gtk \
    --output appimage

ls -l "$LDAI_OUTPUT"
