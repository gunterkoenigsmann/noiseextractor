//! # noiseextractor-core
//!
//! Everything in noiseextractor that does not need a window: audio I/O, the
//! short-time Fourier transform, noise estimation, soft-mask noise reduction,
//! the "find what belongs to the marked part" analysis, the filter model that
//! is saved in project files, and the background engine that processes a
//! recording in chunks, nearest to the user's focus first.
//!
//! The GUI (`crates/gui`) and the command-line tool (`crates/cli`) are thin
//! front ends on top of this crate, so everything that matters can be tested
//! without a display.
//!
//! ## Signal flow
//!
//! ```text
//!  audio ──STFT──► power P(t,k) ──► per filter: soft mask m(t,k) ∈ [0,1]
//!                                      │
//!                         combine masks (see `filter::combine`)
//!                                      ▼
//!          output = ISTFT( X(t,k) · G(t,k) )   for every channel
//! ```
//!
//! All filters produce *soft* masks (gains between 0 and 1) rather than
//! on/off decisions. That keeps the attack and decay of loud sounds and the
//! amplitude ratios inside a sound intact, which is exactly what a hard
//! "remove everything below the noise floor" gate destroys.

// Index loops over (frame, bin) grids read more like the maths they
// implement than iterator chains, so this lint is off crate-wide.
#![allow(clippy::needless_range_loop)]

pub mod analysis;
pub mod audio;
pub mod engine;
pub mod filter;
pub mod noise;
pub mod render;
pub mod stats;
pub mod stft;

pub use audio::AudioData;
pub use engine::{ChunkResult, Engine, EngineConfig, Focus};
pub use filter::{Filter, FilterAction, FilterKind, Project};
pub use stft::StftConfig;
