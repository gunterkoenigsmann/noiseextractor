#!/usr/bin/env bash
# Packs the release binaries into a Debian package for the architecture of
# the machine it runs on (amd64 or arm64):
#
#   packaging/linux/deb.sh VERSION OUTDIR
#
# Expects `cargo build --release -p noiseextractor -p noiseextractor-cli`
# to have run. Writes OUTDIR/noiseextractor_DEBVERSION_ARCH.deb.
#
# Unlike the AppImage, the package bundles no libraries: GTK 3, ALSA and
# the rest come from the system, and `Depends` lists them as computed by
# dpkg-shlibdeps from the binaries. wxWidgets is linked statically by
# wxdragon. The package therefore fits the distribution the binaries were
# built on (Ubuntu 24.04 in CI) and newer ones with the same library
# names, such as Debian 13; the release workflow installs it in clean
# Ubuntu and Debian containers to check that.
#
# Tools: dpkg-dev (dpkg-shlibdeps, dpkg-architecture), librsvg2-bin.
set -euo pipefail

version="$1"
outdir="$(mkdir -p "$2" && cd "$2" && pwd)"
root="$(cd "$(dirname "$0")/../.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-$root/target}"
arch="$(dpkg --print-architecture)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Debian version: a "-" would start the Debian revision, and a pre-release
# must sort before its release, which "~" does (0.2.0~rc1 < 0.2.0). The
# "-1" is the revision of our packaging.
debversion="${version/-/\~}-1"

pkg="$work/pkg"
install -Dm755 "$target_dir/release/noiseextractor" "$pkg/usr/bin/noiseextractor"
install -Dm755 "$target_dir/release/noiseextractor-cli" "$pkg/usr/bin/noiseextractor-cli"
# Debug information makes up most of the binaries; Debian ships them
# stripped (Debian Policy 10.1).
strip --strip-unneeded "$pkg/usr/bin/"*
install -Dm644 "$root/packaging/linux/noiseextractor.desktop" \
    "$pkg/usr/share/applications/noiseextractor.desktop"
install -Dm644 "$root/packaging/icons/noiseextractor.svg" \
    "$pkg/usr/share/icons/hicolor/scalable/apps/noiseextractor.svg"
mkdir -p "$pkg/usr/share/icons/hicolor/256x256/apps"
rsvg-convert -w 256 -h 256 "$root/packaging/icons/noiseextractor.svg" \
    -o "$pkg/usr/share/icons/hicolor/256x256/apps/noiseextractor.png"
# Debian Policy 12.5: the copyright file, with the full license or a
# pointer to /usr/share/common-licenses.
doc="$pkg/usr/share/doc/noiseextractor"
mkdir -p "$doc"
cat > "$doc/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: noiseextractor
Source: https://github.com/gunterkoenigsmann/noiseextractor

Files: *
Copyright: noiseextractor contributors
License: GPL-3.0-or-later
 On Debian systems, the full text of the GNU General Public License
 version 3 can be found in /usr/share/common-licenses/GPL-3.

Files: usr/bin/noiseextractor
Copyright: wxWidgets contributors
License: wxWindows
 The GUI contains wxWidgets (https://wxwidgets.org), linked statically,
 under the wxWindows Library Licence 3.1.
EOF

# dpkg-shlibdeps wants to run inside a source package; a minimal
# debian/control is enough. -O prints the substitution variables.
mkdir -p "$work/src/debian"
printf 'Source: noiseextractor\n\nPackage: noiseextractor\nArchitecture: any\n' \
    > "$work/src/debian/control"
depends="$(cd "$work/src" && dpkg-shlibdeps -O "$pkg/usr/bin/"* | sed -n 's/^shlibs:Depends=//p')"
if [ -z "$depends" ]; then
    echo "error: dpkg-shlibdeps found no dependencies" >&2
    exit 1
fi

mkdir -p "$pkg/DEBIAN"
cat > "$pkg/DEBIAN/control" <<EOF
Package: noiseextractor
Version: $debversion
Architecture: $arch
Maintainer: noiseextractor contributors <https://github.com/gunterkoenigsmann/noiseextractor/issues>
Installed-Size: $(du -sk "$pkg/usr" | cut -f1)
Depends: $depends
Section: sound
Priority: optional
Homepage: https://github.com/gunterkoenigsmann/noiseextractor
Description: mark noises in the spectrogram and suppress or extract them
 noiseextractor shows the spectrogram of a recording. Mark part of a sound
 (say, the fundamental of a bird call) and it finds the overtones,
 resonances and repetitions that belong to it; a list of filters then
 suppresses or extracts them. Background noise is reduced with a soft
 spectral gain that keeps attack, decay and the balance of partials.
 .
 Contains the GUI (noiseextractor) and the command line tool
 (noiseextractor-cli).
EOF

deb="$outdir/noiseextractor_${debversion}_${arch}.deb"
# --root-owner-group: files belong to root, not to the CI user.
dpkg-deb --build --root-owner-group "$pkg" "$deb"
dpkg-deb --info "$deb"
dpkg-deb --contents "$deb"
