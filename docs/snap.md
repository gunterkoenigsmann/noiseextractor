# Snap package

noiseextractor is published on the Snap Store as
[`noiseextractor`](https://snapcraft.io/noiseextractor) (once the steps under
"First upload" below are done). The recipe is
[`snap/snapcraft.yaml`](../snap/snapcraft.yaml); it contains the GUI
(`noiseextractor`) and the command line tool (`noiseextractor.cli`, the
`noiseextractor-cli` binary). Recipe and workflows follow losslessaudiocut's
on purpose; keep the two alike.

## What gets built where

| Workflow | When | What |
|---|---|---|
| [`snap.yml`](../.github/workflows/snap.yml) | every push to `main` (a newer push cancels the running build); PRs that touch `snap/` or the workflow; by hand | builds the snap, installs it, denoises `packaging/smoke-test.wav` with the CLI and starts the GUI with it under Xvfb. Artifacts `snap-amd64` and `snap-arm64`. Never uploads. |
| [`snap-publish.yml`](../.github/workflows/snap-publish.yml) | Mondays 04:17 UTC; by hand | builds the newest `main` and uploads it to **edge**, builds the newest GitHub release and uploads it to **stable** (skipped while there is no release, or when the release has no snap recipe). Artifacts `snap-edge-amd64`, `snap-edge-arm64` and, with a release, `snap-stable-…`. |

### Where the `.snap` files are

Every run keeps what it built for 90 days: open the run (Actions → *Snap*
or *Snap publish* → the run), scroll to the **Artifacts** section at the
bottom of its summary page and download the zip, which contains the
`.snap`. A run that is still going, or one whose build step failed, has
none yet. The weekly job's first scheduled run is the first Monday after
it was added; until then only runs started by hand or by a push exist.

`snapcore/action-build` and `action-publish` still target Node.js 20, so
GitHub prints a deprecation warning and runs them on Node.js 24. That is
harmless; we keep them, since snapcore maintains them.

The weekly rebuild is what brings security fixes to users: the snap bundles
ALSA and its PulseAudio plugin from the Ubuntu 24.04 archive, and those only
update when the snap is rebuilt. GTK and the rest of the desktop stack come
from the `gnome-46-2404` snap, which the store updates on its own. Rust
crates are pinned by `Cargo.lock` and change when a PR updates them (the
weekly `cargo-update.yml` proposes that).

### Version and grade

Set by the recipe from git:

- a commit with a tag `vX.Y.Z` builds version `X.Y.Z`, grade `stable`;
- any other commit builds e.g. `0.1.0+git20261117.7bec07a` (version from
  `Cargo.toml`, commit date and hash), grade `devel`.

The store only accepts grade `stable` on the candidate and stable channels,
so a release needs a tag (and a GitHub release, which is what the weekly job
looks for). v0.1.0 predates the recipe, so stable stays empty until the
next release; the weekly job notes that and only publishes edge.

## First upload (by hand)

The store needs the name registered and one upload by its owner before CI
may publish.

1. Register the name (once, with the Ubuntu One account that should own the
   snap):
   ```sh
   sudo snap install snapcraft --classic
   snapcraft login
   snapcraft register noiseextractor
   ```
2. Get a `.snap`: download the `snap-amd64` (or `snap-arm64`) artifact of
   the latest [Snap workflow run](https://github.com/gunterkoenigsmann/noiseextractor/actions/workflows/snap.yml)
   on `main` and unzip it, or build it yourself with
   `sudo snap install snapcraft --classic` and `snapcraft pack` in the
   repository (needs LXD: `sudo snap install lxd && sudo lxd init --auto`).
3. Try it locally before uploading:
   ```sh
   sudo snap install --dangerous ./noiseextractor_*_amd64.snap   # or _arm64
   noiseextractor             # open a file, mark a sound, play it, save
   noiseextractor.cli --help
   ```
   Things only a desktop can check: sound comes out (PulseAudio or
   PipeWire), the file dialogs work, and files on USB sticks open after
   `sudo snap connect noiseextractor:removable-media`.
4. Upload it to edge:
   ```sh
   sudo snap install snapcraft --classic   # if not done yet
   snapcraft login
   snapcraft upload --release=edge noiseextractor_*_amd64.snap
   snapcraft upload --release=edge noiseextractor_*_arm64.snap
   ```
   The first upload of a new snap may wait for a manual store review; the
   upload command prints the status, and the same is shown on
   <https://snapcraft.io/noiseextractor/releases>.
5. Fill in the store listing at
   <https://snapcraft.io/noiseextractor/listing>: category (Audio & Video),
   screenshots (`docs/screenshot.png` is a start), and check icon and
   description. Summary, description, license and links come from
   `snapcraft.yaml`.

## Credentials for the weekly job

CI uploads with a store login that only the name's owner can create. It is
limited to this one snap and to uploading and releasing.

1. Export the login (it asks for the store password and 2FA):
   ```sh
   snapcraft export-login \
     --snaps=noiseextractor \
     --channels=edge,stable \
     --acls=package_access,package_push,package_update,package_release \
     --expires="$(date -d '+1 year' +%Y-%m-%d)" \
     snapcraft-credentials.txt
   ```
2. Add the file's content as a repository secret named
   `SNAPCRAFT_STORE_CREDENTIALS`: on GitHub,
   [Settings → Secrets and variables → Actions → New repository secret](https://github.com/gunterkoenigsmann/noiseextractor/settings/secrets/actions/new),
   paste the whole file, or with the GitHub CLI:
   ```sh
   gh secret set SNAPCRAFT_STORE_CREDENTIALS \
     -R gunterkoenigsmann/noiseextractor < snapcraft-credentials.txt
   ```
3. Delete the file: `rm snapcraft-credentials.txt`.
4. Start the workflow once by hand (Actions → Snap publish → Run workflow)
   and check that edge got a new revision.

The login expires after the date given above; the weekly job then fails at
the publish step, and steps 1 to 3 renew it. Without the secret the job
still builds, but only warns instead of uploading.

## Making a release

1. Make the release as [docs/release.md](release.md) says: pushing the tag
   `vX.Y.Z` creates the GitHub release.
2. Run *Snap publish* by hand, or wait for Monday: the release goes to
   stable, `main` to edge. (Pre-releases such as `v1.2.0-rc1` are not
   GitHub's "latest release" and never reach stable.)
