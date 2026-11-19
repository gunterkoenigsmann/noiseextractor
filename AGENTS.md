# Notes for coding agents

## Rules of this project (from the maintainer)

* New features go on a feature branch and land through a pull request.
  PRs may be merged once **all** CI checks (Linux, Windows, macOS) pass.
* Every function, type and field gets a doc comment; code that is not
  self-explanatory gets a comment explaining *why*. CI enforces this with
  `missing_docs` and `clippy::missing_docs_in_private_items` as errors.
* Every change comes with automated tests. Prefer testing logic in
  `crates/core` (headless) or through the CLI (`crates/cli/tests`).
* Found a bug you are not fixing right now? File a GitHub issue.

## Layout

* `crates/core`: all signal processing and state that does not need a window.
  Start reading at `src/lib.rs` (signal flow), `filter.rs` (how filters
  combine), `engine.rs` (chunked background processing), `discover.rs`
  (finding a recording's sounds by NMF; found sounds are fitted frame by
  frame, so chunks stay seamless).
* Real recordings for tests live in `crates/core/tests/fixtures` (excerpts
  of the maintainer's own recordings, GPL; keep them to a few hundred kB
  and list them in the README there).
* `crates/cli`: thin front end; `tests/cli.rs` holds end-to-end tests.
* `crates/gui`: wxDragon front end with AUI panes (`panes.rs`). The view
  models are pure and unit-tested (`view.rs`, `overview.rs`, `scrub.rs`,
  `detection.rs`, the `Log` in `log.rs`, `Declick` in `player.rs`,
  `RecentFiles` in `recent.rs`); keep
  logic out of event handlers where possible. The toolbar (`toolbar.rs`) only re-sends
  menu command ids, so give a new toolbar button a menu item first; its
  icons are inline SVGs with a light and a dark palette. Status messages go through
  `report()` so they also reach the log pane. Scrubbing and overview follow
  losslessaudiocut (gunterkoenigsmann/losslessaudiocut) on purpose; keep
  them alike.

## Releases

Pushing a tag `vX.Y.Z` builds, smoke-tests and publishes Debian packages,
AppImages, Windows zips/installers and macOS DMGs
(`.github/workflows/release.yml`, scripts in `packaging/`, steps in
`docs/release.md`). The snap (`snap/snapcraft.yaml`, steps in
`docs/snap.md`) is built by `snap.yml` on every push to main and rebuilt
and uploaded to edge (main) and stable (latest release) every Monday by
`snap-publish.yml`. Adding an input format means adding it to `FILE_TYPES`
in `crates/core/src/audio.rs` and to the packages' file associations (the
snap's desktop file in `snap/gui` too); `crates/core/tests/packaging.rs`
fails until they agree.

## Commands

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace          # builds wxWidgets the first time (minutes)
cargo test                      # core + CLI only, fast
```

On Linux the GUI needs `libgtk-3-dev` and `libasound2-dev`.

## Pitfalls

* The STFT frame grid is global (frame i is centred on sample i·hop). Chunked
  processing relies on this to be seamless; do not introduce per-chunk grids.
  The same holds for the adaptive noise estimate: it is taken on a global
  grid of frames (`noise::NoiseReduction::noise_track`) and interpolated,
  so never estimate the noise from "the current chunk".
* GUI state lives in an `Rc<RefCell<State>>`. GTK can deliver paint events
  synchronously while a handler holds the borrow, so the paint handler uses
  `try_borrow` and handlers request repaints via `request_repaint()` instead of
  calling `refresh` directly. Never hold a borrow across a modal dialog.
* The audio callback must never block: it uses `Engine::try_chunk`.
* A project file names its recording (`Project::audio`, saved as
  `"recording"`, relative to the project file where possible, see
  `stored_recording` in `crates/core/src/filter.rs`). Anything that saves
  a project for a recording must set it; the GUI does so in
  `install_audio`, the CLI before `--save-project`.
