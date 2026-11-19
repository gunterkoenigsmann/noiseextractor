//! Storage for decoded samples that does not have to fit into RAM.
//!
//! A whole night recorded in stereo at 48 kHz is 8 h · 3600 s · 48000 ·
//! 2 channels · 4 bytes ≈ 11 GB of 32-bit floats, plus the mono mix the
//! analysis works on. Kept on the heap that is more than most computers
//! have, and the program used to be killed for running out of memory.
//!
//! So a recording whose samples would take more than
//! [`StorageOptions::spill_above_bytes`] is decoded into anonymous scratch
//! files instead, one per channel, which are then mapped into memory. The
//! program still sees an ordinary `&[f32]`, but the pages belong to the
//! operating system's file cache: it reads them in when they are used and
//! drops them again when memory gets tight, instead of killing the program.
//! Short recordings (tests, clips) stay on the heap and never touch the disk.
//!
//! The scratch files live in the user's cache directory, not in the temp
//! directory, because on many Linux systems `/tmp` is a `tmpfs`, i.e. RAM.
//! They are deleted by the operating system as soon as the program closes
//! them or ends, even after a crash.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Where and from what size on decoded audio goes to scratch files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageOptions {
    /// Recordings whose channels and mono mix together would take more
    /// bytes than this are stored in mapped scratch files.
    pub spill_above_bytes: usize,
    /// Directory for the scratch files.
    pub dir: PathBuf,
}

impl Default for StorageOptions {
    /// 256 MB on the heap at most (about 23 minutes of stereo at 48 kHz),
    /// scratch files in [`scratch_dir`].
    fn default() -> Self {
        Self {
            spill_above_bytes: 256 << 20,
            dir: scratch_dir(),
        }
    }
}

/// The directory for scratch files: `noiseextractor` in the user's cache
/// directory (`$XDG_CACHE_HOME` or `~/.cache` on Linux, `~/Library/Caches`
/// on macOS, `%LOCALAPPDATA%` on Windows), or the temp directory if there is
/// none or it cannot be created.
pub fn scratch_dir() -> PathBuf {
    let env = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        env("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library").join("Caches"))
    } else {
        env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|h| h.join(".cache")))
    };
    base.map(|b| b.join("noiseextractor"))
        .filter(|d| std::fs::create_dir_all(d).is_ok())
        .unwrap_or_else(std::env::temp_dir)
}

/// One channel of samples, either on the heap or in a mapped scratch file.
/// Dereferences to `[f32]`; cloning is cheap (the storage is shared).
#[derive(Clone)]
pub struct Samples(Repr);

/// The two kinds of storage behind [`Samples`].
#[derive(Clone)]
enum Repr {
    /// Ordinary memory.
    Heap(Arc<Vec<f32>>),
    /// A read-only mapping of a scratch file holding native-endian `f32`s.
    Mapped(Arc<memmap2::Mmap>),
}

impl Samples {
    /// Whether the samples live in a mapped scratch file.
    pub fn is_mapped(&self) -> bool {
        matches!(self.0, Repr::Mapped(_))
    }
}

impl From<Vec<f32>> for Samples {
    /// Wraps heap samples without copying them.
    fn from(v: Vec<f32>) -> Self {
        Samples(Repr::Heap(Arc::new(v)))
    }
}

impl std::ops::Deref for Samples {
    type Target = [f32];

    /// The samples as a slice.
    fn deref(&self) -> &[f32] {
        match &self.0 {
            Repr::Heap(v) => v,
            // A mapping starts on a page boundary, so it is aligned for
            // f32, and the file only ever received whole f32s (see
            // `SamplesWriter`), so the cast cannot fail.
            Repr::Mapped(m) => bytemuck::cast_slice(&m[..]),
        }
    }
}

impl AsRef<[f32]> for Samples {
    /// The samples as a slice.
    fn as_ref(&self) -> &[f32] {
        self
    }
}

impl PartialEq for Samples {
    /// Equal if the samples are, wherever they are stored.
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl PartialEq<Vec<f32>> for Samples {
    /// Compares with plain samples.
    fn eq(&self, other: &Vec<f32>) -> bool {
        **self == other[..]
    }
}

impl std::fmt::Debug for Samples {
    /// Shows where the samples are and how many there are, not the
    /// (possibly billions of) samples themselves.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = if self.is_mapped() { "mapped" } else { "heap" };
        write!(f, "Samples({} {kind})", self.len())
    }
}

impl<'a> IntoIterator for &'a Samples {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;

    /// Iterates over the samples.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Collects one channel's samples, on the heap until [`SamplesWriter::spill`]
/// moves them into a scratch file.
pub struct SamplesWriter(WriterRepr);

/// The two states of a [`SamplesWriter`].
enum WriterRepr {
    /// Still collecting in memory.
    Heap(Vec<f32>),
    /// Writing to a scratch file; `usize` counts the samples written.
    File(BufWriter<File>, usize),
}

impl SamplesWriter {
    /// A writer collecting on the heap, with room for `capacity` samples.
    pub fn new(capacity: usize) -> Self {
        SamplesWriter(WriterRepr::Heap(Vec::with_capacity(capacity)))
    }

    /// A writer that goes straight to a scratch file in `dir`.
    pub fn in_file(dir: &Path) -> Result<Self> {
        let mut w = Self::new(0);
        w.spill(dir)?;
        Ok(w)
    }

    /// Number of samples so far.
    pub fn len(&self) -> usize {
        match &self.0 {
            WriterRepr::Heap(v) => v.len(),
            WriterRepr::File(_, n) => *n,
        }
    }

    /// True if no sample has been written.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the samples go to a scratch file.
    pub fn is_spilled(&self) -> bool {
        matches!(self.0, WriterRepr::File(..))
    }

    /// Appends one sample.
    pub fn push(&mut self, s: f32) -> Result<()> {
        match &mut self.0 {
            WriterRepr::Heap(v) => v.push(s),
            WriterRepr::File(w, n) => {
                w.write_all(&s.to_ne_bytes()).context(DISK_FULL)?;
                *n += 1;
            }
        }
        Ok(())
    }

    /// Appends several samples.
    pub fn extend(&mut self, s: &[f32]) -> Result<()> {
        match &mut self.0 {
            WriterRepr::Heap(v) => v.extend_from_slice(s),
            WriterRepr::File(w, n) => {
                w.write_all(bytemuck::cast_slice(s)).context(DISK_FULL)?;
                *n += s.len();
            }
        }
        Ok(())
    }

    /// Moves the samples collected so far into a new scratch file in `dir`;
    /// later ones go there too. Does nothing if that already happened.
    pub fn spill(&mut self, dir: &Path) -> Result<()> {
        if let WriterRepr::Heap(v) = &mut self.0 {
            let file = tempfile::tempfile_in(dir)
                .with_context(|| format!("cannot create a scratch file in {}", dir.display()))?;
            let mut w = BufWriter::with_capacity(1 << 20, file);
            w.write_all(bytemuck::cast_slice(v)).context(DISK_FULL)?;
            let n = v.len();
            self.0 = WriterRepr::File(w, n);
        }
        Ok(())
    }

    /// The finished channel.
    pub fn finish(self) -> Result<Samples> {
        match self.0 {
            WriterRepr::Heap(mut v) => {
                // Growing by doubling can leave up to half the capacity unused.
                v.shrink_to_fit();
                Ok(v.into())
            }
            // Mapping an empty file fails on some systems.
            WriterRepr::File(_, 0) => Ok(Vec::new().into()),
            WriterRepr::File(w, _) => {
                let file = w.into_inner().map_err(|e| e.into_error()).context(DISK_FULL)?;
                // SAFETY: the file is an anonymous scratch file that only
                // this process has a handle to (`tempfile` unlinks it at
                // once on Unix and opens it for exclusive use, deleted on
                // close, on Windows), and the handle is moved into the
                // mapping's owner and dropped here, so nothing can change
                // or truncate it while it is mapped.
                let map = unsafe { memmap2::Mmap::map(&file) }.context("cannot map a scratch file")?;
                Ok(Samples(Repr::Mapped(Arc::new(map))))
            }
        }
    }
}

/// Context for write errors to scratch files, which are nearly always a
/// full disk.
const DISK_FULL: &str = "cannot write decoded audio to the scratch file (is the disk full?)";

/// The average of `channels` (all equally long). One channel is returned as
/// it is (shared, not copied); a mix of mapped channels is mapped as well.
pub fn mix(channels: &[Samples], options: &StorageOptions) -> Result<Samples> {
    match channels {
        [] => Ok(Vec::new().into()),
        [one] => Ok(one.clone()),
        _ => {
            let len = channels[0].len();
            let mut out = if channels.iter().any(Samples::is_mapped) {
                SamplesWriter::in_file(&options.dir)?
            } else {
                SamplesWriter::new(len)
            };
            let n = channels.len() as f32;
            // Blocks keep the buffer small and the reads sequential.
            const BLOCK: usize = 1 << 16;
            let mut buf = vec![0.0f32; BLOCK];
            for start in (0..len).step_by(BLOCK) {
                let end = (start + BLOCK).min(len);
                let b = &mut buf[..end - start];
                b.fill(0.0);
                for c in channels {
                    for (o, &s) in b.iter_mut().zip(&c[start..end]) {
                        *o += s / n;
                    }
                }
                out.extend(b)?;
            }
            out.finish()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spilled_samples_read_back_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let data: Vec<f32> = (0..100_000).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut w = SamplesWriter::new(0);
        w.extend(&data[..1000]).unwrap();
        w.spill(dir.path()).unwrap();
        assert!(w.is_spilled());
        for &s in &data[1000..1010] {
            w.push(s).unwrap();
        }
        w.extend(&data[1010..]).unwrap();
        assert_eq!(w.len(), data.len());
        let s = w.finish().unwrap();
        assert!(s.is_mapped());
        assert_eq!(s, data);
        // Nothing is left behind once the samples are gone (on Windows the
        // file is visible until it is closed, on Unix it never is).
        drop(s);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn empty_spilled_channel_is_fine() {
        let dir = tempfile::tempdir().unwrap();
        let s = SamplesWriter::in_file(dir.path()).unwrap().finish().unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn mix_averages_and_shares_a_single_channel() {
        let a: Samples = vec![1.0, 0.0, 0.5].into();
        let b: Samples = vec![0.0, 1.0, 0.5].into();
        let opts = StorageOptions::default();
        assert_eq!(mix(&[a.clone(), b], &opts).unwrap(), vec![0.5, 0.5, 0.5]);
        let one = mix(std::slice::from_ref(&a), &opts).unwrap();
        assert_eq!(one.as_ptr(), a.as_ptr(), "a single channel is not copied");
    }

    #[test]
    fn mix_of_mapped_channels_is_mapped() {
        let dir = tempfile::tempdir().unwrap();
        let opts = StorageOptions {
            spill_above_bytes: 0,
            dir: dir.path().to_path_buf(),
        };
        let mapped = |v: &[f32]| {
            let mut w = SamplesWriter::in_file(dir.path()).unwrap();
            w.extend(v).unwrap();
            w.finish().unwrap()
        };
        // Longer than one mixing block, to cover the block loop's seams.
        let n = 200_000;
        let a: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let m = mix(&[mapped(&a), mapped(&b)], &opts).unwrap();
        assert!(m.is_mapped());
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x / 2.0 + y / 2.0).collect();
        assert_eq!(m, want);
    }

    #[test]
    fn scratch_dir_exists() {
        assert!(scratch_dir().is_dir());
    }
}
