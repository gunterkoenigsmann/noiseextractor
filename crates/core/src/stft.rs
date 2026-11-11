//! Short-time Fourier transform on a *global* frame grid.
//!
//! Frame `i` is centred on sample `i · hop` of the recording, no matter which
//! part of the recording is being analysed. Because every chunk of work uses
//! the same grid, chunks can be processed independently (and in any order)
//! and still overlap-add to exactly the same output as processing the whole
//! file in one go. Samples outside the recording are treated as silence.
//!
//! Analysis and synthesis both use a periodic Hann window; the overlap-add
//! output is normalised by the summed squared window, which gives perfect
//! reconstruction for any hop ≤ fft_size / 2 when the spectrum is untouched.

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Size and spacing of the STFT frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StftConfig {
    /// FFT length in samples (and window length). Must be even.
    pub fft_size: usize,
    /// Distance between frame centres in samples.
    pub hop: usize,
}

impl Default for StftConfig {
    /// 2048-point frames with 75 % overlap: ~43 ms at 48 kHz, a common
    /// compromise between time and frequency resolution for speech and birds.
    fn default() -> Self {
        Self { fft_size: 2048, hop: 512 }
    }
}

impl StftConfig {
    /// Number of frequency bins (DC up to and including Nyquist).
    pub fn bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Number of frames needed so that every sample of a signal of
    /// `len` samples is covered by at least one frame centre region.
    pub fn frame_count(&self, len: usize) -> usize {
        len.div_ceil(self.hop) + 1
    }

    /// Centre frequency of `bin` in Hz.
    pub fn bin_to_hz(&self, bin: f32, sample_rate: u32) -> f32 {
        bin * sample_rate as f32 / self.fft_size as f32
    }

    /// Fractional bin index of `hz`.
    pub fn hz_to_bin(&self, hz: f32, sample_rate: u32) -> f32 {
        hz * self.fft_size as f32 / sample_rate as f32
    }

    /// Frames on each side of a frame whose windows overlap it; i.e. the
    /// padding a chunk needs so its interior overlap-adds completely.
    pub fn overlap_frames(&self) -> usize {
        (self.fft_size / 2).div_ceil(self.hop)
    }

    /// Checks that the configuration can reconstruct a signal.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.fft_size >= 16 && self.fft_size.is_multiple_of(2), "fft_size must be even and >= 16");
        anyhow::ensure!(self.hop >= 1 && self.hop <= self.fft_size / 2, "hop must be in 1..=fft_size/2");
        Ok(())
    }
}

/// Complex spectra of consecutive frames `first_frame ..`.
#[derive(Clone, Debug)]
pub struct Spectra {
    /// Global index of the first frame.
    pub first_frame: i64,
    /// One `Vec` of `bins` complex values per frame.
    pub frames: Vec<Vec<Complex32>>,
}

impl Spectra {
    /// Power |X|² of every bin of every frame.
    pub fn power(&self) -> Vec<Vec<f32>> {
        self.frames.iter().map(|f| f.iter().map(|c| c.norm_sqr()).collect()).collect()
    }
}

/// Reusable forward/inverse FFT plans plus the window for one configuration.
///
/// Creating plans is comparatively expensive, so worker threads keep one
/// `Stft` each and reuse it for every chunk.
pub struct Stft {
    cfg: StftConfig,
    window: Vec<f32>,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
}

impl Stft {
    /// Plans the FFTs for `cfg`. Panics if the configuration is invalid
    /// (call [`StftConfig::validate`] on user input first).
    pub fn new(cfg: StftConfig) -> Self {
        cfg.validate().expect("invalid STFT configuration");
        let mut planner = RealFftPlanner::<f32>::new();
        let n = cfg.fft_size;
        // Periodic Hann window: sums to a constant under overlap-add.
        let window = (0..n)
            .map(|i| {
                let s = (std::f32::consts::PI * i as f32 / n as f32).sin();
                s * s
            })
            .collect();
        Self { cfg, window, forward: planner.plan_fft_forward(n), inverse: planner.plan_fft_inverse(n) }
    }

    /// The configuration this transform was planned for.
    pub fn config(&self) -> StftConfig {
        self.cfg
    }

    /// Transforms frames `first_frame .. first_frame + count` of `signal`.
    /// Frame indices may be negative or reach past the end; missing samples
    /// are zeros.
    pub fn analyze(&self, signal: &[f32], first_frame: i64, count: usize) -> Spectra {
        let n = self.cfg.fft_size;
        let half = (n / 2) as i64;
        let mut input = self.forward.make_input_vec();
        let mut scratch = self.forward.make_scratch_vec();
        let mut frames = Vec::with_capacity(count);
        for f in 0..count {
            let centre = (first_frame + f as i64) * self.cfg.hop as i64;
            let start = centre - half;
            for (i, slot) in input.iter_mut().enumerate() {
                let idx = start + i as i64;
                let x = if idx >= 0 && (idx as usize) < signal.len() { signal[idx as usize] } else { 0.0 };
                *slot = x * self.window[i];
            }
            let mut out = self.forward.make_output_vec();
            self.forward
                .process_with_scratch(&mut input, &mut out, &mut scratch)
                .expect("FFT buffer sizes are fixed by the plan");
            frames.push(out);
        }
        Spectra { first_frame, frames }
    }

    /// Overlap-adds `spectra` and returns output samples
    /// `out_start .. out_start + out_len`.
    ///
    /// The result is only exact where *all* frames overlapping a sample are
    /// contained in `spectra`; callers pad by [`StftConfig::overlap_frames`].
    /// The normalisation uses the window sum of the full global grid, so the
    /// borders of a padded chunk are not boosted.
    pub fn synthesize(&self, spectra: &Spectra, out_start: i64, out_len: usize) -> Vec<f32> {
        let n = self.cfg.fft_size;
        let half = (n / 2) as i64;
        let mut out = vec![0.0f32; out_len];
        let mut buf = self.inverse.make_input_vec();
        let mut time = self.inverse.make_output_vec();
        let mut scratch = self.inverse.make_scratch_vec();
        let scale = 1.0 / n as f32; // realfft's inverse is unnormalised.
        for (f, spec) in spectra.frames.iter().enumerate() {
            let centre = (spectra.first_frame + f as i64) * self.cfg.hop as i64;
            let start = centre - half;
            // Skip frames that do not touch the requested output range.
            if start + n as i64 <= out_start || start >= out_start + out_len as i64 {
                continue;
            }
            buf.copy_from_slice(spec);
            // The imaginary parts of DC and Nyquist must be zero for a real
            // inverse; masking keeps them real but rounding might not.
            buf[0].im = 0.0;
            let last = buf.len() - 1;
            buf[last].im = 0.0;
            self.inverse
                .process_with_scratch(&mut buf, &mut time, &mut scratch)
                .expect("FFT buffer sizes are fixed by the plan");
            for i in 0..n {
                let idx = start + i as i64 - out_start;
                if idx >= 0 && (idx as usize) < out_len {
                    out[idx as usize] += time[i] * scale * self.window[i];
                }
            }
        }
        let norm = self.ola_norm();
        for (i, s) in out.iter_mut().enumerate() {
            let pos = (out_start + i as i64).rem_euclid(self.cfg.hop as i64) as usize;
            *s /= norm[pos];
        }
        out
    }

    /// Σ w²(t) over all frames, as a function of the position within a hop.
    /// On the global grid this is periodic with period `hop`.
    fn ola_norm(&self) -> Vec<f32> {
        let n = self.cfg.fft_size;
        let hop = self.cfg.hop;
        let half = n / 2;
        (0..hop)
            .map(|p| {
                // Sample at offset p from a frame centre sees window index
                // half + p - j·hop for every frame j that covers it.
                let mut s = 0.0;
                let mut k: i64 = -((n / hop) as i64) - 1;
                while k <= (n / hop) as i64 + 1 {
                    let idx = half as i64 + p as i64 - k * hop as i64;
                    if idx >= 0 && (idx as usize) < n {
                        let w = self.window[idx as usize];
                        s += w * w;
                    }
                    k += 1;
                }
                s.max(1e-9)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signal(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32 * 0.031).sin() + 0.3 * (i as f32 * 0.17).cos()) * 0.5).collect()
    }

    #[test]
    fn unmodified_spectra_reconstruct_the_signal() {
        let cfg = StftConfig { fft_size: 256, hop: 64 };
        let stft = Stft::new(cfg);
        let x = test_signal(3000);
        let spectra = stft.analyze(&x, -(cfg.overlap_frames() as i64), cfg.frame_count(x.len()) + 2 * cfg.overlap_frames());
        let y = stft.synthesize(&spectra, 0, x.len());
        for (i, (a, b)) in x.iter().zip(&y).enumerate() {
            assert!((a - b).abs() < 1e-4, "sample {i}: {a} vs {b}");
        }
    }

    #[test]
    fn chunked_synthesis_matches_whole_signal() {
        // Processing two halves with padding must give the same samples as
        // processing everything at once: the basis of background chunking.
        let cfg = StftConfig { fft_size: 128, hop: 32 };
        let stft = Stft::new(cfg);
        let x = test_signal(2048);
        let pad = cfg.overlap_frames() as i64;
        let split_frame = 32i64; // sample 1024
        let first = stft.analyze(&x, -pad, (split_frame + 2 * pad) as usize);
        let second = stft.analyze(&x, split_frame - pad, (64 - split_frame + 2 * pad) as usize);
        let mut y = stft.synthesize(&first, 0, 1024);
        y.extend(stft.synthesize(&second, 1024, 1024));
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn sine_lands_in_expected_bin() {
        let cfg = StftConfig { fft_size: 512, hop: 128 };
        let stft = Stft::new(cfg);
        let sr = 8000;
        let hz = 1000.0;
        let x: Vec<f32> = (0..4096).map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sr as f32).sin()).collect();
        let p = stft.analyze(&x, 10, 1).power();
        let peak = p[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(peak, cfg.hz_to_bin(hz, sr).round() as usize);
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(StftConfig { fft_size: 1023, hop: 256 }.validate().is_err());
        assert!(StftConfig { fft_size: 1024, hop: 0 }.validate().is_err());
        assert!(StftConfig { fft_size: 1024, hop: 600 }.validate().is_err());
        assert!(StftConfig::default().validate().is_ok());
    }
}
