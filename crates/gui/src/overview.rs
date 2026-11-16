//! The waveform overview: the whole recording squeezed into one strip,
//! with a frame showing which part the spectrogram shows. Dragging the
//! frame (or clicking beside it) scrolls the spectrogram there. The same
//! idea as losslessaudiocut's minimap.
//!
//! [`Peaks`] holds the minimum and maximum of every block of samples,
//! computed once when the file is loaded, so drawing the strip for an hour
//! of audio only touches a few hundred thousand numbers. [`Overview`] is the
//! strip's view model (pixel ↔ sample, frame, dragging). Both are free of
//! GUI types so they can be tested.

/// Samples per block of [`Peaks`].
pub const BLOCK: usize = 256;

/// Narrowest frame drawn, in pixels, so that a deep zoom stays visible.
pub const MIN_FRAME_WIDTH: f64 = 3.0;

/// Minimum and maximum per block of [`BLOCK`] samples of a signal.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Peaks {
    /// `(min, max)` of each block, in order.
    blocks: Vec<(f32, f32)>,
}

impl Peaks {
    /// Computes the block peaks of `signal`.
    pub fn new(signal: &[f32]) -> Self {
        let blocks = signal
            .chunks(BLOCK)
            .map(|b| {
                b.iter()
                    .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
            })
            .collect();
        Peaks { blocks }
    }

    /// `(min, max)` for each of `width` pixel columns spanning the whole
    /// signal. A column narrower than a block shows that block.
    pub fn columns(&self, width: u32) -> Vec<(f32, f32)> {
        let n = self.blocks.len();
        if n == 0 || width == 0 {
            return vec![(0.0, 0.0); width as usize];
        }
        (0..width as usize)
            .map(|c| {
                let a = c * n / width as usize;
                let b = ((c + 1) * n / width as usize).clamp(a + 1, n);
                self.blocks[a.min(n - 1)..b]
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(lo, hi), &(l, h)| (lo.min(l), hi.max(h)))
            })
            .collect()
    }
}

/// The overview strip of a recording of `total` samples, `width` pixels
/// wide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Overview {
    /// Samples in the recording.
    total: usize,
    /// Width of the strip in pixels.
    width: u32,
    /// While the button is held: how far after the spectrogram's start the
    /// pointer grabbed it (samples), so the frame does not jump under the
    /// pointer.
    grab: Option<usize>,
}

impl Overview {
    /// The overview of `total` samples in `width` pixels.
    pub fn new(total: usize, width: u32) -> Self {
        Overview {
            total,
            width: width.max(1),
            grab: None,
        }
    }

    /// The strip was resized.
    pub fn set_width(&mut self, width: u32) {
        self.width = width.max(1);
    }

    /// Pixel column of sample `s`.
    pub fn sample_to_x(&self, s: usize) -> f64 {
        s as f64 * self.width as f64 / self.total.max(1) as f64
    }

    /// Sample under pixel column `x`, clamped to the recording.
    pub fn x_to_sample(&self, x: f64) -> usize {
        ((x.max(0.0) * self.total as f64 / self.width as f64) as usize).min(self.total)
    }

    /// The frame around `visible` (the spectrogram's visible samples) as
    /// pixel columns `left..right`, at least [`MIN_FRAME_WIDTH`] wide.
    pub fn frame(&self, visible: (usize, usize)) -> (f64, f64) {
        let (l, r) = (self.sample_to_x(visible.0), self.sample_to_x(visible.1));
        let grow = (MIN_FRAME_WIDTH - (r - l)).max(0.0) / 2.0;
        (l - grow, r + grow)
    }

    /// Button pressed at pixel `x` while the spectrogram shows `visible`.
    /// Returns where the spectrogram should start: unchanged when the frame
    /// was grabbed, else centred on the pointer.
    pub fn press(&mut self, x: f64, visible: (usize, usize)) -> usize {
        let pos = self.x_to_sample(x);
        let (l, r) = self.frame(visible);
        let grab = if (l..r).contains(&x) {
            pos.saturating_sub(visible.0)
        } else {
            (visible.1 - visible.0) / 2
        };
        self.grab = Some(grab);
        pos.saturating_sub(grab)
    }

    /// The pointer moved to pixel `x`. While the button is held, returns
    /// where the spectrogram should start (the viewport clamps it).
    pub fn drag(&mut self, x: f64) -> Option<usize> {
        let grab = self.grab?;
        Some(self.x_to_sample(x).saturating_sub(grab))
    }

    /// Button released.
    pub fn release(&mut self) {
        self.grab = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_cover_every_column() {
        // 10 blocks: block i holds values -i .. i.
        let signal: Vec<f32> = (0..10 * BLOCK)
            .map(|n| {
                let i = (n / BLOCK) as f32;
                if n % 2 == 0 {
                    i
                } else {
                    -i
                }
            })
            .collect();
        let p = Peaks::new(&signal);
        assert_eq!(
            p.columns(5),
            vec![(-1.0, 1.0), (-3.0, 3.0), (-5.0, 5.0), (-7.0, 7.0), (-9.0, 9.0)]
        );
        // More columns than blocks: each column shows its block.
        let wide = p.columns(20);
        assert_eq!(wide.len(), 20);
        assert_eq!((wide[0], wide[19]), ((0.0, 0.0), (-9.0, 9.0)));
        assert_eq!(Peaks::new(&[]).columns(3), vec![(0.0, 0.0); 3]);
    }

    /// 1000 samples over 100 pixels: 10 samples per pixel.
    fn strip() -> Overview {
        Overview::new(1000, 100)
    }

    #[test]
    fn the_frame_shows_the_visible_part() {
        let o = strip();
        assert_eq!(o.frame((200, 500)), (20.0, 50.0));
        assert_eq!(o.frame((0, 1000)), (0.0, 100.0));
        // Two samples are a fifth of a pixel: widened around the middle.
        let (l, r) = o.frame((500, 502));
        assert!((r - l - MIN_FRAME_WIDTH).abs() < 1e-9);
        assert!(((l + r) / 2.0 - 50.1).abs() < 1e-9);
    }

    #[test]
    fn grabbing_the_frame_moves_it_with_the_pointer() {
        let mut o = strip();
        assert_eq!(o.press(30.0, (200, 500)), 200, "nothing moves yet");
        assert_eq!(o.drag(45.0), Some(350));
        assert_eq!(o.drag(0.0), Some(0), "not before the beginning");
        o.release();
        assert_eq!(o.drag(60.0), None);
    }

    #[test]
    fn clicking_beside_the_frame_centres_it_there() {
        let mut o = strip();
        assert_eq!(o.press(80.0, (200, 500)), 650);
        assert_eq!(o.drag(90.0), Some(750));
        assert_eq!(o.press(5.0, (200, 500)), 0, "clamped at the start");
    }

    #[test]
    fn resizing_rescales() {
        let mut o = strip();
        o.set_width(200);
        assert_eq!(o.frame((200, 500)), (40.0, 100.0));
        assert_eq!(o.x_to_sample(1000.0), 1000, "clamped at the end");
    }
}
