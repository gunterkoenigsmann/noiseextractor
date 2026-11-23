# The Python package

`pip install noiseextractor` installs a Python API and the command line
tool `noiseextractor-cli`, both on top of the same Rust code as the GUI.
It exists for people and programs that can install a Python package but
not a desktop program: scripts, notebooks, bioacoustics pipelines, and AI
assistants, which look for and install packages from PyPI when someone
asks them to remove or extract noise. The user documentation is the
package's README, [`crates/python/README.md`](../crates/python/README.md),
which PyPI shows; its wording and the keywords in `pyproject.toml` are
what such searches find, so keep them plain ("remove background noise",
"extract a bird song").

## Layout

| Path | What |
|---|---|
| `crates/python/src/lib.rs` | The extension module `noiseextractor._native` (PyO3). Thin: plain Python values in and out, float32 sample buffers as `bytes` (no NumPy needed on the Rust side), the work in `noiseextractor-core` and `noiseextractor-cli`'s library with the interpreter lock released. |
| `crates/python/python/noiseextractor/` | The API users see: `denoise`, `process`, `mark`, `discover`, `spectrogram`, `new_project`, `load`, `denoise_array`, `process_array`, the result dataclasses, and the `noiseextractor-cli` entry point (`_cli.py`, which runs `noiseextractor_cli::run` in-process). |
| `crates/python/pyproject.toml` | Package metadata for maturin. The version comes from `Cargo.toml`. |
| `crates/python/LICENSE`, `crates/python/COPYING.LESSER` | Copies of the repository's license files (license files must lie inside the package directory); CI checks that they agree. The package is LGPL-3.0-or-later, like `crates/core` and `crates/cli`. |
| `crates/python/tests/` | pytest tests, run against an installed wheel. |
| `.github/workflows/python.yml` | Builds, tests and publishes the wheels and the sdist. |

The wheels need no system libraries: the C encoders (LAME, libvorbis,
libopus) are compiled from vendored sources as for the desktop packages,
and the package never links the sound output (that is only the GUI's).

## Building locally

With Rust, a C compiler and CMake:

```sh
cd crates/python
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release       # or: maturin build --release
cd /tmp && python -m pytest ~/noiseextractor/crates/python/tests
```

Run pytest from outside the source tree, or the source package (without
the native module) shadows the installed one. `cargo build --workspace`
builds the extension module too, without linking libpython (PyO3's
`extension-module` feature).

## Releasing

A release tag `vX.Y.Z` builds and tests all wheels. Uploading them to
PyPI as version `X.Y.Z` needs a one-time setup by the owner of the PyPI
project (done for v0.2.0), after which every release tag publishes:

1. On pypi.org, *Your account*, *Publishing*: add a pending trusted
   publisher of type GitHub with project name `noiseextractor`, owner
   `gunterkoenigsmann`, repository `noiseextractor`, workflow `python.yml`,
   environment `pypi`. (After the first upload it shows under the
   project's *Publishing* settings.)
2. On GitHub, *Settings*, *Environments*: create the environment `pypi`
   (optionally with required reviewers, so that each upload waits for a
   click).

No token is stored anywhere: PyPI trusts the workflow through GitHub's
OIDC token.

To upload a release whose upload failed or was skipped: *Actions*,
*Python package*, *Run workflow*, with `tag` set to the release tag (e.g.
`v0.2.0`) and *publish* ticked. It builds and tests that tag's code with
the current workflow and uploads it; files already on PyPI are skipped.
(Re-running the tag's old run would use the workflow as it was then.)
