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
  combine), `engine.rs` (chunked background processing).
* `crates/cli`: thin front end; `tests/cli.rs` holds end-to-end tests.
* `crates/gui`: wxDragon front end. `view.rs` is pure and unit-tested; keep
  logic out of event handlers where possible.

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
* GUI state lives in an `Rc<RefCell<State>>`. GTK can deliver paint events
  synchronously while a handler holds the borrow, so the paint handler uses
  `try_borrow` and handlers request repaints via `request_repaint()` instead of
  calling `refresh` directly. Never hold a borrow across a modal dialog.
* The audio callback must never block: it uses `Engine::try_chunk`.
