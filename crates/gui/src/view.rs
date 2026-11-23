//! Pure viewport arithmetic of the spectrogram panel: zooming, scrolling and
//! converting mouse drags into selections (rectangles, or free-form
//! outlines drawn with [`SelectTool::FreeForm`]). Kept free of GUI types so
//! it can be unit-tested.

use noiseextractor_core::analysis::Selection;
use noiseextractor_core::render::View;

/// Smallest zoom: one sample per pixel is far beyond the STFT resolution,
/// zooming in further would only magnify blocks.
const MIN_SAMPLES_PER_PX: f64 = 1.0;
/// Lowest upper frequency bound reachable by frequency zoom.
const MIN_HIGH_HZ: f32 = 500.0;

/// How a drag on the spectrogram marks a region (toolbar / Filter menu).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelectTool {
    /// The drag spans a rectangle from where it started to where it ends.
    #[default]
    Rectangle,
    /// The pointer draws the outline of the region ("lasso"), which is
    /// closed by a straight line back to the start when the button is
    /// released. Suits sounds that are not rectangular in the
    /// spectrogram: a U-shaped sweep, a slanted chirp, a call whose
    /// overtones a rectangle would cut.
    FreeForm,
}

/// Smallest distance in pixels between two recorded points of a free-form
/// stroke; closer ones add nothing visible and only bloat the outline.
const STROKE_MIN_STEP_PX: f32 = 2.0;
/// How far (pixels) the simplified outline of a stroke may deviate from
/// what was drawn: less than the width of the drawn line.
const STROKE_TOLERANCE_PX: f32 = 1.0;
/// A drag whose points all stay within this many pixels of its start is a
/// click (it moves the play head), whichever tool is active.
pub const CLICK_SLOP_PX: f32 = 3.0;

/// The pointer's path while a free-form selection is being drawn, in
/// pixels of the spectrogram panel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stroke {
    /// Recorded points, at least [`STROKE_MIN_STEP_PX`] apart.
    points: Vec<(f32, f32)>,
}

impl Stroke {
    /// A stroke starting at `p`.
    pub fn new(p: (f32, f32)) -> Self {
        Self { points: vec![p] }
    }

    /// Adds the pointer position `p`, unless it is too close to the last
    /// recorded point to matter.
    pub fn push(&mut self, p: (f32, f32)) {
        match self.points.last() {
            Some(&last) if distance(last, p) < STROKE_MIN_STEP_PX => {}
            _ => self.points.push(p),
        }
    }

    /// The recorded points.
    pub fn points(&self) -> &[(f32, f32)] {
        &self.points
    }

    /// Whether the stroke never left the start by more than
    /// [`CLICK_SLOP_PX`]: then it was meant as a click.
    pub fn is_click(&self) -> bool {
        let Some(&first) = self.points.first() else {
            return true;
        };
        self.points
            .iter()
            .all(|&p| (p.0 - first.0).abs() < CLICK_SLOP_PX && (p.1 - first.1).abs() < CLICK_SLOP_PX)
    }
}

/// Euclidean distance between two pixel positions.
fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// Distance of `p` from the line segment `a`–`b`.
fn segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return distance(p, a);
    }
    let u = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0);
    distance(p, (a.0 + u * dx, a.1 + u * dy))
}

/// Ramer–Douglas–Peucker: the fewest of `points` (first and last always
/// kept) such that the polyline through them stays within `tolerance` of
/// every dropped point. A hand-drawn stroke of hundreds of points becomes
/// a few dozen corners, which keeps project files small and the analysis'
/// inside test fast.
pub fn simplify(points: &[(f32, f32)], tolerance: f32) -> Vec<(f32, f32)> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    // Explicit stack instead of recursion: a long stroke cannot overflow it.
    let mut spans = vec![(0, points.len() - 1)];
    while let Some((a, b)) = spans.pop() {
        let farthest = (a + 1..b)
            .map(|i| (i, segment_distance(points[i], points[a], points[b])))
            .max_by(|x, y| x.1.total_cmp(&y.1));
        if let Some((i, d)) = farthest {
            if d > tolerance {
                keep[i] = true;
                spans.push((a, i));
                spans.push((i, b));
            }
        }
    }
    points.iter().zip(keep).filter_map(|(&p, k)| k.then_some(p)).collect()
}

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

    /// Shows samples `start..end` across a panel `width` pixels wide, as far
    /// as the zoom limits and the recording allow. A span too short for the
    /// deepest zoom stays centred.
    pub fn show_range(&mut self, start: usize, end: usize, width: u32) {
        let (a, b) = (start.min(end) as f64, start.max(end) as f64);
        let w = width.max(1) as f64;
        self.view.samples_per_px = ((b - a) / w).clamp(MIN_SAMPLES_PER_PX, self.max_spp(width));
        self.view.start_sample = (a + b) / 2.0 - self.view.samples_per_px * w / 2.0;
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
        Selection::rect(t(x0), t(x1), f(y0), f(y1))
    }

    /// Converts a free-form stroke (pixel points, see [`Stroke`]) into a
    /// selection enclosed by it, clamped like [`Self::drag_to_selection`].
    /// The stroke is simplified first (see [`simplify`]). `None` if it
    /// encloses no area, e.g. a straight line.
    pub fn stroke_to_selection(&self, stroke: &[(f32, f32)], height: u32) -> Option<Selection> {
        let sr = self.sample_rate as f64;
        let outline = simplify(stroke, STROKE_TOLERANCE_PX)
            .into_iter()
            .map(|(x, y)| {
                (
                    self.sample_at(x) as f64 / sr,
                    self.view.y_to_hz(y.clamp(0.0, height as f32 - 1.0), height),
                )
            })
            .collect();
        Selection::free_form(outline)
    }

    /// Pixel corners of the outline of `sel` for drawing as a polygon: the
    /// free-form outline, or the rectangle's four corners.
    pub fn selection_outline(&self, sel: &Selection, height: u32) -> Vec<(i32, i32)> {
        let sr = self.sample_rate as f64;
        let px = |(t, hz): (f64, f32)| {
            (
                self.view.sample_to_x(t * sr).round() as i32,
                self.view.hz_to_y(hz, height).round() as i32,
            )
        };
        if sel.is_free_form() {
            sel.outline.iter().map(|&p| px(p)).collect()
        } else {
            [
                (sel.start_s, sel.low_hz),
                (sel.end_s, sel.low_hz),
                (sel.end_s, sel.high_hz),
                (sel.start_s, sel.high_hz),
            ]
            .into_iter()
            .map(px)
            .collect()
        }
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
    fn show_range_zooms_to_the_span() {
        let mut v = Viewport::fit(48_000, 48_000, 1000);
        v.show_range(30_000, 10_000, 1000);
        assert_eq!(v.visible_range(1000), (10_000, 30_000));
        // Shorter than the deepest zoom allows: centred on the span.
        v.show_range(20_000, 20_100, 1000);
        assert_eq!(v.visible_range(1000), (19_550, 20_550));
        // At the end of the recording: clamped inside it.
        v.show_range(47_900, 48_000, 1000);
        assert_eq!(v.visible_range(1000), (47_000, 48_000));
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
    fn stroke_skips_tiny_moves_and_knows_a_click() {
        let mut s = Stroke::new((10.0, 10.0));
        s.push((10.5, 10.5));
        s.push((11.0, 11.0));
        assert_eq!(s.points().len(), 1, "moves below the step are dropped");
        assert!(s.is_click());
        s.push((10.0, 40.0));
        assert_eq!(s.points().len(), 2);
        assert!(!s.is_click());
    }

    #[test]
    fn simplify_keeps_corners_and_drops_points_on_straight_lines() {
        // An L drawn with many points: the corner and both ends stay.
        let mut pts: Vec<(f32, f32)> = (0..=20).map(|i| (i as f32, 0.0)).collect();
        pts.extend((1..=20).map(|i| (20.0, i as f32 + 0.3 * (i % 2) as f32)));
        assert_eq!(simplify(&pts, 1.0), vec![(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]);
        assert_eq!(simplify(&pts[..2], 1.0), pts[..2].to_vec());
    }

    #[test]
    fn stroke_becomes_free_form_selection_and_back() {
        let v = Viewport::fit(10_000, 10_000, 100); // 100 samples per px, 0.01 s per px
                                                    // A triangle drawn with intermediate points along its edges.
        let mut s = Stroke::new((10.0, 90.0));
        for i in 1..=40 {
            s.push((10.0 + i as f32, 90.0 - 2.0 * i as f32)); // up to (50, 10)
        }
        for i in 1..=40 {
            s.push((50.0 + i as f32, 10.0 + 2.0 * i as f32)); // down to (90, 90)
        }
        let sel = v.stroke_to_selection(s.points(), 100).expect("encloses an area");
        assert!(sel.is_free_form());
        assert_eq!(sel.outline.len(), 3, "straight edges simplified: {:?}", sel.outline);
        assert!((sel.start_s - 0.1).abs() < 1e-9 && (sel.end_s - 0.9).abs() < 1e-9);
        // Inside near the base, outside beside the tip.
        let hz = |y: f32| v.view.y_to_hz(y, 100);
        assert!(sel.contains(0.5, hz(80.0)));
        assert!(!sel.contains(0.15, hz(20.0)));
        assert_eq!(v.selection_outline(&sel, 100), vec![(10, 90), (50, 10), (90, 90)]);
        // A straight line encloses nothing.
        let line = [(10.0, 50.0), (50.0, 50.0), (90.0, 50.0)];
        assert!(v.stroke_to_selection(&line, 100).is_none());
    }

    #[test]
    fn rectangle_outline_has_four_corners() {
        let v = Viewport::fit(10_000, 10_000, 100);
        let sel = v.drag_to_selection((10.0, 90.0), (50.0, 10.0), 100);
        let corners = v.selection_outline(&sel, 100);
        assert_eq!(corners.len(), 4);
        assert_eq!((corners[0].0, corners[1].0), (10, 50));
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
