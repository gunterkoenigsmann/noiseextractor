#!/usr/bin/env bash
# Prints the version that release packages are named after, and checks that
# a release tag matches the workspace version in Cargo.toml.
#
#   packaging/version.sh [REF]
#
# REF is the git ref the build is for (in CI: $GITHUB_REF), default HEAD.
#
# - refs/tags/vX.Y.Z[-pre]: prints X.Y.Z[-pre]. The part before any "-" must
#   equal `version` in [workspace.package] of Cargo.toml, otherwise the
#   script fails: a release whose binaries report another version than its
#   name is worse than no release. (A "-rc1" style suffix marks a GitHub
#   pre-release, see .github/workflows/release.yml.)
# - anything else: prints a development version such as
#   0.1.0-dev.20261104.b46df06 (Cargo version, commit date, short hash).
#
# The snap recipe (snap/snapcraft.yaml) derives its version from the same
# tag the same way, so a release has one version number everywhere.
set -euo pipefail

ref="${1:-HEAD}"
root="$(cd "$(dirname "$0")/.." && pwd)"

# First `version = "..."` in the root Cargo.toml: the one in
# [workspace.package], which every crate inherits.
cargo_version="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$root/Cargo.toml" | head -n 1)"
if [ -z "$cargo_version" ]; then
    echo "error: no version in $root/Cargo.toml" >&2
    exit 1
fi

case "$ref" in
refs/tags/v*)
    version="${ref#refs/tags/v}"
    # Strip a pre-release suffix (1.2.0-rc1 -> 1.2.0) for the comparison.
    if [ "${version%%-*}" != "$cargo_version" ]; then
        echo "error: tag v$version does not match version $cargo_version in Cargo.toml" >&2
        exit 1
    fi
    echo "$version"
    ;;
*)
    sha="$(git -C "$root" rev-parse --short=7 HEAD)"
    date="$(git -C "$root" log -1 --format=%cd --date=format:%Y%m%d)"
    echo "${cargo_version}-dev.${date}.${sha}"
    ;;
esac
