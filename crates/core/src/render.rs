//! Turning chunk results into spectrogram pixels.
//!
//! Colour code (as requested in the project brief):
//!
//! * **blue** – the signal that remains after all enabled filters,
//! * **green** – energy inside the area the user marked,
//! * **red** – energy the analysis believes belongs to the marked sound,
//! * **orange** – marked by the user *and* found by the analysis.
//!
//! Overlays are weighted by the *original* level, so they show which energy a
//! filter grabs even when that filter has already removed it from the blue
//! layer.

use crate::engine::{u8_to_db, ChunkLayout, ChunkResult};
use std::sync::Arc;

/// RGB colours of the layers.
pub const BLUE: [f32; 3] = [60.0, 125.0, 255.0];
/// Colour of the user's selection.
pub const GREEN: [f32; 3] = [50.0, 220.0, 70.0];
/// Colour of inferred membership.
pub const RED: [f32; 3] = [235.0, 45.0, 45.0];
/// Colour where selection and inference overlap.
pub const ORANGE: [f32; 3] = [255.0, 160.0, 0.0];
/// Background where a chunk has not been processed yet.
pub const PENDING: [u8; 3] = [28, 28, 34];

/// Which part of the time-frequency plane to draw, and how.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// First sample at the left edge.
    pub start_sample: f64,
    /// Samples per horizontal pixel.
    pub samples_per_px: f64,
    /// Frequency at the bottom edge.
    pub low_hz: f32,
    /// Frequency at the top edge.
    pub high_hz: f32,
    /// Logarithmic instead of linear frequency axis.
    pub log_freq: bool,
    /// Level drawn as black.
    pub db_min: f32,
    /// Level drawn at full brightness.
    pub db_max: f32,
}

impl View {
    /// Frequency at pixel row `y` of an image `height` pixels tall (row 0
    /// is the top).
    pub fn y_to_hz(&self, y: f32, height: u32) -> f32 {
        let frac = 1.0 - (y + 0.5) / height.max(1) as f32;
        if self.log_freq {
            let lo = self.low_hz.max(10.0).ln();
            let hi = self.high_hz.max(self.low_hz.max(10.0) + 1.0).ln();
            (lo + (hi - lo) * frac).exp()
        } else {
            self.low_hz + (self.high_hz - self.low_hz) * frac
        }
    }

    /// Inverse of [`View::y_to_hz`].
    pub fn hz_to_y(&self, hz: f32, height: u32) -> f32 {
        let frac = if self.log_freq {
            let lo = self.low_hz.max(10.0).ln();
            let hi = self.high_hz.max(self.low_hz.max(10.0) + 1.0).ln();
            (hz.max(1.0).ln() - lo) / (hi - lo)
        } else {
            (hz - self.low_hz) / (self.high_hz - self.low_hz)
        };
        (1.0 - frac) * height as f32 - 0.5
    }

    /// Sample at pixel column `x`.
    pub fn x_to_sample(&self, x: f32) -> f64 {
        self.start_sample + x as f64 * self.samples_per_px
    }

    /// Pixel column of `sample`.
    pub fn sample_to_x(&self, sample: f64) -> f32 {
        ((sample - self.start_sample) / self.samples_per_px) as f32
    }
}

/// Mixes the layer levels of one time-frequency bin into a colour.
/// `out`, `orig`: levels in dBFS; `sel`, `inf`: masks in 0..=1.
pub fn pixel(view: &View, out: f32, orig: f32, sel: f32, inf: f32) -> [u8; 3] {
    let norm = |db: f32| ((db - view.db_min) / (view.db_max - view.db_min)).clamp(0.0, 1.0);
    let b = norm(out);
    let o = norm(orig);
    let s = sel * o;
    let i = inf * o;
    let both = s.min(i);
    let (s, i) = (s - both, i - both);
    let mut c = [0.0f32; 3];
    for ch in 0..3 {
        // Overlays replace the blue where they are strong, so they stay
        // visible on top of loud material.
        let overlay = s * GREEN[ch] + i * RED[ch] + both * ORANGE[ch];
        let cover = (s + i + both).min(1.0);
        c[ch] = b * BLUE[ch] * (1.0 - cover) + overlay;
    }
    c.map(|v| v.round().clamp(0.0, 255.0) as u8)
}

/// Renders an RGB image (`width × height × 3` bytes, rows top to bottom).
///
/// `chunk` returns the latest result for a chunk index, if any. Where a
/// pixel column spans several frames, the loudest one is shown so short
/// events stay visible when zoomed out (at most 8 frames are inspected per
/// column to keep drawing fast).
pub fn render_rgb(
    view: &View,
    width: u32,
    height: u32,
    layout: ChunkLayout,
    sample_rate: u32,
    fft_size: usize,
    chunk: &dyn Fn(usize) -> Option<Arc<ChunkResult>>,
) -> Vec<u8> {
    let mut img = vec![0u8; width as usize * height as usize * 3];
    let bins = fft_size / 2 + 1;
    // Bin for every row, computed once.
    let row_bins: Vec<usize> = (0..height)
        .map(|y| {
            let hz = view.y_to_hz(y as f32, height);
            ((hz * fft_size as f32 / sample_rate as f32).round().max(0.0) as usize).min(bins - 1)
        })
        .collect();
    let mut cached: Option<Arc<ChunkResult>> = None;
    for x in 0..width {
        let s0 = view.x_to_sample(x as f32);
        let s1 = view.x_to_sample(x as f32 + 1.0);
        if s1 <= 0.0 || s0 >= layout.total_samples as f64 {
            continue; // outside the recording: leave black
        }
        // Frames whose centres fall into this column.
        let f0 = (s0.max(0.0) / layout.hop as f64).round() as i64;
        let f1 = ((s1 / layout.hop as f64).round() as i64).max(f0 + 1);
        let step = ((f1 - f0) / 8).max(1);
        let mut frames = Vec::new();
        let mut f = f0;
        while f < f1 {
            let c = (f as usize / layout.chunk_frames).min(layout.chunk_count() - 1);
            if cached.as_ref().is_none_or(|r| r.index != c) {
                cached = chunk(c);
            }
            if let Some(r) = &cached {
                let i = f - r.first_frame;
                if i >= 0 && (i as usize) < r.frames {
                    frames.push((r.clone(), i as usize));
                }
            }
            f += step;
        }
        for (y, &k) in row_bins.iter().enumerate() {
            let at = (y * width as usize + x as usize) * 3;
            if frames.is_empty() {
                img[at..at + 3].copy_from_slice(&PENDING);
                continue;
            }
            // loudest of the inspected frames
            let (r, i) = frames
                .iter()
                .max_by_key(|(r, i)| r.original_db[i * r.bins + k])
                .expect("non-empty");
            let idx = i * r.bins + k;
            let px = pixel(
                view,
                u8_to_db(r.output_db[idx]),
                u8_to_db(r.original_db[idx]),
                r.selected[idx] as f32 / 255.0,
                r.inferred[idx] as f32 / 255.0,
            );
            img[at..at + 3].copy_from_slice(&px);
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> View {
        View {
            start_sample: 0.0,
            samples_per_px: 1.0,
            low_hz: 0.0,
            high_hz: 1000.0,
            log_freq: false,
            db_min: -100.0,
            db_max: 0.0,
        }
    }

    #[test]
    fn colours_follow_the_legend() {
        let v = view();
        let blue = pixel(&v, 0.0, 0.0, 0.0, 0.0);
        assert!(blue[2] > 200 && blue[0] < 100);
        let green = pixel(&v, 0.0, 0.0, 1.0, 0.0);
        assert!(green[1] > 200 && green[0] < 100 && green[2] < 100);
        let red = pixel(&v, -100.0, 0.0, 0.0, 1.0);
        assert!(red[0] > 200 && red[1] < 100);
        let orange = pixel(&v, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(orange, [255, 160, 0]);
        assert_eq!(pixel(&v, -100.0, -100.0, 1.0, 1.0), [0, 0, 0]);
    }

    #[test]
    fn axis_mappings_are_inverse() {
        for log_freq in [false, true] {
            let v = View {
                low_hz: 50.0,
                high_hz: 8000.0,
                log_freq,
                ..view()
            };
            for y in [0.0, 10.0, 99.0] {
                let hz = v.y_to_hz(y, 100);
                assert!((v.hz_to_y(hz, 100) - y).abs() < 1e-2);
            }
        }
        let v = View {
            start_sample: 100.0,
            samples_per_px: 2.5,
            ..view()
        };
        assert!((v.sample_to_x(v.x_to_sample(7.0)) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn unprocessed_chunks_are_drawn_as_pending() {
        let layout = ChunkLayout {
            hop: 4,
            chunk_frames: 4,
            total_samples: 64,
        };
        let img = render_rgb(
            &View {
                samples_per_px: 4.0,
                ..view()
            },
            4,
            2,
            layout,
            2000,
            8,
            &|_| None,
        );
        assert_eq!(&img[0..3], &PENDING);
    }
}
