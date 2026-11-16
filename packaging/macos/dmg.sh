#!/usr/bin/env bash
# Builds noiseextractor.app and packs it into a disk image for the
# architecture of the Mac it runs on (arm64 or x86_64):
#
#   packaging/macos/dmg.sh VERSION OUTDIR
#
# Expects `cargo build --release -p noiseextractor -p noiseextractor-cli`
# to have run, and Homebrew's `librsvg` (rsvg-convert) to be installed.
# Writes OUTDIR/noiseextractor-VERSION-macos-ARCH.dmg.
#
# Layout of the bundle:
#
#   noiseextractor.app/Contents/
#     Info.plist
#     MacOS/noiseextractor        the GUI (CFBundleExecutable)
#     MacOS/noiseextractor-cli    the command line tool
#     Resources/noiseextractor.icns
#
# Both programs only link system frameworks (wxWidgets is built in
# statically), so unlike losslessaudiocut no libraries are bundled; the
# script fails if that ever changes, see "Binaries" below.
#
# The app is not signed with a Developer ID nor notarized (see
# docs/release.md); it only gets the ad-hoc signature that Apple silicon
# needs to run any code at all.
set -euo pipefail

version="$1"
outdir="$(mkdir -p "$2" && cd "$2" && pwd)"
root="$(cd "$(dirname "$0")/../.." && pwd)"
arch="$(uname -m)"            # arm64 or x86_64
target_dir="${CARGO_TARGET_DIR:-$root/target}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

app="$work/dmg/noiseextractor.app"
contents="$app/Contents"
mkdir -p "$contents/MacOS" "$contents/Resources"

# --- Info.plist --------------------------------------------------------------
# macOS wants up to three dot-separated integers in the bundle versions, so
# a pre-release or development suffix is cut off there (it stays in the
# "Get Info" string and the file name).
short_version="${version%%-*}"
sed -e "s/@VERSION@/$version/g" -e "s/@SHORT_VERSION@/$short_version/g" \
    "$root/packaging/macos/Info.plist" > "$contents/Info.plist"
plutil -lint "$contents/Info.plist"

# --- Icon --------------------------------------------------------------------
# iconutil builds an .icns from an .iconset folder with these fixed names.
iconset="$work/noiseextractor.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256 512; do
    rsvg-convert -w "$size" -h "$size" "$root/packaging/icons/noiseextractor.svg" \
        -o "$iconset/icon_${size}x${size}.png"
    rsvg-convert -w $((size * 2)) -h $((size * 2)) "$root/packaging/icons/noiseextractor.svg" \
        -o "$iconset/icon_${size}x${size}@2x.png"
done
iconutil -c icns -o "$contents/Resources/noiseextractor.icns" "$iconset"

# --- Binaries ------------------------------------------------------------------
cp "$target_dir/release/noiseextractor" "$target_dir/release/noiseextractor-cli" "$contents/MacOS/"
# Only system libraries may be linked: a Homebrew library does not exist on
# a user's Mac, and the app would only fail there, not here. Should a
# dependency start pulling one in, bundle it with dylibbundler as
# losslessaudiocut does for FFmpeg.
leaks=0
for f in "$contents/MacOS/"*; do
    otool -L "$f"
    if otool -L "$f" | tail -n +2 | grep -vE '^\s*(/usr/lib/|/System/Library/)'; then
        echo "error: $f links a library that is not part of macOS" >&2
        leaks=1
    fi
done
[ "$leaks" -eq 0 ]

# Apple silicon refuses to run unsigned code; the linker signs the binaries
# ad hoc, but the bundle needs a signature of its own (covering Info.plist
# and the resources) for Gatekeeper's checks to treat it as one app.
codesign --force --sign - "$contents/MacOS/noiseextractor-cli"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"

# --- Disk image ----------------------------------------------------------------
# The usual drag-to-install layout: the app next to a link to /Applications.
ln -s /Applications "$work/dmg/Applications"
dmg="$outdir/noiseextractor-$version-macos-$arch.dmg"
hdiutil create -volname "noiseextractor $version" -srcfolder "$work/dmg" \
    -ov -format UDZO "$dmg"
ls -l "$dmg"
