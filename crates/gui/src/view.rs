//! Pure viewport arithmetic of the spectrogram panel: zooming, scrolling and
//! converting mouse drags into selections. Kept free of GUI types so it can
//! be unit-tested.

use noiseextractor_core::analysis::Selection;
use noiseextractor_core::render::View;

/// Smallest zoom: one sample per pixel is far beyond the STFT resolution,
/// zooming in further would only magnify blocks.
const MIN_SAMPLES_PER_PX: f64 = 1.0;
/// Lowest upper frequency bound reachable by frequency zoom.
const MIN_HIGH_HZ: f32 = 500.0;

/// The visible window onto the recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Rendering parameters shared with the core renderer.
    pub view: View,
    /// Samples in the recording.
    pub total_samples: usize,
    /// Sample rate of the recording.
    pub sample_rate: u32,
}

impl Viewport {
    /// Shows the whole recording, all frequencies, in `width` pixels.
    pub fn fit(total_samples: usize, sample_rate: u32, width: u32) -> Self {
        let view = View {
            start_sample: 0.0,
            samples_per_px: (total_samples.max(1) as f64 / width.max(1) as f64).max(MIN_SAMPLES_PER_PX),
            low_hz: 0.0,
            high_hz: sample_rate as f32 / 2.0,
            log_freq: false,
            db_min: -100.0,
            db_max: -10.0,
        };
        Self {
            view,
            total_samples,
            sample_rate,
        }
    }

    /// Largest useful samples-per-pixel value: the whole file fits.
    fn max_spp(&self, width: u32) -> f64 {
        (self.total_samples.max(1) as f64 / width.max(1) as f64).max(MIN_SAMPLES_PER_PX)
    }

    /// Keeps the view inside the recording.
    pub fn clamp(&mut self, width: u32) {
        let max_spp = self.max_spp(width);
        let v = &mut self.view;
        v.samples_per_px = v.samples_per_px.clamp(MIN_SAMPLES_PER_PX, max_spp);
        let max_start = (self.total_samples as f64 - v.samples_per_px * width as f64).max(0.0);
        v.start_sample = v.start_sample.clamp(0.0, max_start);
    }

    /// Zooms the time axis by `factor` (< 1 zooms in) keeping the sample
    /// under pixel column `anchor_x` in place.
    pub fn zoom_time(&mut self, anchor_x: f32, factor: f64, width: u32) {
        let anchor = self.view.x_to_sample(anchor_x);
        self.view.samples_per_px = (self.view.samples_per_px * factor).clamp(MIN_SAMPLES_PER_PX, self.max_spp(width));
        self.view.start_sample = anchor - anchor_x as f64 * self.view.samples_per_px;
        self.clamp(width);
    }

    /// Scrolls by `dx` pixels (positive = later in time).
    pub fn scroll(&mut self, dx: f32, width: u32) {
        self.view.start_sample += dx as f64 * self.view.samples_per_px;
        self.clamp(width);
    }

    /// Scales the upper frequency bound by `factor`, between 500 Hz and
    /// Nyquist.
    pub fn zoom_freq(&mut self, factor: f32) {
        let nyquist = self.sample_rate as f32 / 2.0;
        self.view.high_hz = (self.view.high_hz * factor).clamp(MIN_HIGH_HZ.min(nyquist), nyquist);
    }

    /// Switches between linear and logarithmic frequency axis.
    pub fn toggle_log(&mut self) {
        self.view.log_freq = !self.view.log_freq;
        // A log axis cannot start at 0 Hz.
        self.view.low_hz = if self.view.log_freq { 50.0 } else { 0.0 };
    }

    /// Visible sample range for a panel `width` pixels wide.
    pub fn visible_range(&self, width: u32) -> (usize, usize) {
        let a = self.view.start_sample.max(0.0) as usize;
        let b = (self.view.start_sample + self.view.samples_per_px * width as f64).max(0.0) as usize;
        (a.min(self.total_samples), b.min(self.total_samples))
    }

    /// Sample under pixel column `x`, clamped to the recording.
    pub fn sample_at(&self, x: f32) -> usize {
        (self.view.x_to_sample(x).max(0.0) as usize).min(self.total_samples)
    }

    /// Converts a drag from `(x0, y0)` to `(x1, y1)` into a selection,
    /// clamped to the recording and the visible frequencies.
    pub fn drag_to_selection(&self, (x0, y0): (f32, f32), (x1, y1): (f32, f32), height: u32) -> Selection {
        let sr = self.sample_rate as f64;
        let t = |x: f32| self.sample_at(x) as f64 / sr;
        let f = |y: f32| self.view.y_to_hz(y.clamp(0.0, height as f32 - 1.0), height);
        Selection {
            start_s: t(x0),
            end_s: t(x1),
            low_hz: f(y0),
            high_hz: f(y1),
        }
        .normalized()
    }

    /// Pixel rectangle `(x, y, w, h)` of `sel` for drawing.
    pub fn selection_rect(&self, sel: &Selection, height: u32) -> (i32, i32, i32, i32) {
        let sr = self.sample_rate as f64;
        let x0 = self.view.sample_to_x(sel.start_s * sr);
        let x1 = self.view.sample_to_x(sel.end_s * sr);
        let y0 = self.view.hz_to_y(sel.high_hz, height);
        let y1 = self.view.hz_to_y(sel.low_hz, height);
        (
            x0 as i32,
            y0 as i32,
            (x1 - x0).max(1.0) as i32,
            (y1 - y0).max(1.0) as i32,
        )
    }
}

/// Formats seconds as `m:ss.mmm` (or `h:mm:ss` for long recordings).
pub fn format_time(seconds: f64) -> String {
    let s = seconds.max(0.0);
    if s >= 3600.0 {
        format!(
            "{}:{:02}:{:02}",
            (s / 3600.0) as u64,
            ((s % 3600.0) / 60.0) as u64,
            (s % 60.0) as u64
        )
    } else {
        format!("{}:{:06.3}", (s / 60.0) as u64, s % 60.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_shows_everything_and_clamps() {
        let mut v = Viewport::fit(48_000, 48_000, 1000);
        assert_eq!(v.visible_range(1000), (0, 48_000));
        v.scroll(500.0, 1000);
        assert_eq!(v.view.start_sample, 0.0, "cannot scroll past the end when fitted");
    }

    #[test]
    fn zoom_keeps_anchor_in_place() {
        let mut v = Viewport::fit(480_000, 48_000, 1000);
        let before = v.view.x_to_sample(250.0);
        v.zoom_time(250.0, 0.5, 1000);
        assert!((v.view.x_to_sample(250.0) - before).abs() < 1e-6);
        assert!((v.view.samples_per_px - 240.0).abs() < 1e-9);
        // Zooming out beyond the whole file is capped.
        v.zoom_time(0.0, 1000.0, 1000);
        assert_eq!(v.visible_range(1000), (0, 480_000));
    }

    #[test]
    fn drag_becomes_normalized_selection() {
        let v = Viewport::fit(10_000, 10_000, 100); // 100 samples per px
        let sel = v.drag_to_selection((50.0, 99.0), (10.0, 0.0), 100);
        assert!((sel.start_s - 0.1).abs() < 1e-9 && (sel.end_s - 0.5).abs() < 1e-9);
        assert!(sel.low_hz < 50.0 && sel.high_hz > 4900.0);
        let (x, _, w, _) = v.selection_rect(&sel, 100);
        assert_eq!((x, w), (10, 40));
    }

    #[test]
    fn freq_zoom_and_log_toggle_stay_in_range() {
        let mut v = Viewport::fit(1000, 16_000, 100);
        v.zoom_freq(0.01);
        assert_eq!(v.view.high_hz, 500.0);
        v.zoom_freq(100.0);
        assert_eq!(v.view.high_hz, 8000.0);
        v.toggle_log();
        assert!(v.view.log_freq && v.view.low_hz > 0.0);
    }

    #[test]
    fn time_formatting() {
        assert_eq!(format_time(61.5), "1:01.500");
        assert_eq!(format_time(3725.0), "1:02:05");
    }
}
