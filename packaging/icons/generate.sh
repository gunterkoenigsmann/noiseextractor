#!/usr/bin/env bash
# Regenerates noiseextractor.ico (Windows program icon and installer) from
# noiseextractor.svg. The .ico is committed because the Windows runners have
# no SVG renderer; run this after changing the SVG.
#
# Needs rsvg-convert (librsvg2-bin) and ImageMagick (`convert`).
set -euo pipefail
cd "$(dirname "$0")"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
pngs=()
for s in 16 24 32 48 64 128 256; do
    rsvg-convert -w "$s" -h "$s" noiseextractor.svg -o "$tmp/$s.png"
    pngs+=("$tmp/$s.png")
done
# One .ico with every size; Windows picks the best one for each place.
convert "${pngs[@]}" noiseextractor.ico
