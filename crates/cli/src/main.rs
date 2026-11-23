//! `noiseextractor-cli`: command line front end of noiseextractor.
//! Everything but the process exit lives in the library (`src/lib.rs`),
//! which the Python package shares.

fn main() {
    std::process::exit(noiseextractor_cli::run(std::env::args_os()));
}
