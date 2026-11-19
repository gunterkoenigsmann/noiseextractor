# noiseextractor

Interactively mark noises in the spectrogram of an audio file, then suppress or
extract them, together with everything the program finds to belong to them.

![noiseextractor: a marked bird call (green), its overtones and its repetitions found by the analysis (red)](docs/screenshot.png)

## What it does

* **Spectrogram with a colour code**: blue is the signal that remains after all
  active filters, green is the area you marked, red is what the analysis
  believes belongs to the marked sound, orange is both.
* **Mark a part of a sound, find the rest.** Drag a rectangle around, say, the
  fundamental of a bird call. The analysis tracks the pitch inside the
  rectangle frame by frame and looks for partials that follow it at rational
  ratios (overtones, and sub-harmonics if you marked an overtone), so a
  drifting pitch is no problem. Resonances that do not move with the pitch
  but swell and fade together with it are found by common modulation.
  A sound without a pitch (insect pulses, a hiss, rustling) is recognised as
  a noise band: then no partials are searched, the marked band itself is the
  sound, and the analysis reports its extent instead of a pitch.
* **Find the same sound elsewhere.** The marked sound, together with
  everything found to belong to it, becomes a spectro-temporal template. It
  is matched against the whole recording (normalised cross-correlation,
  tolerating a small frequency shift), so marking one bird call also catches
  its repetitions. Each repetition gets the same soft mask, weighted by how
  well it matches and by how far it stands above the noise there.
* **Pulsed sounds are found by their rhythm.** If the marked sound is a
  pulse train (a cricket, a katydid, a trill, a machine), the search
  compares the pulse rate (the spectrum of the band's loudness over time)
  and which part of the band is busy, not the exact pattern. A burst that is
  20 dB quieter, longer or shorter, or partly covered by another sound is
  still recognised, and each found burst is masked over its whole length.
  The *Loudness needed* slider sets how far above the noise a burst has to
  be.
* **Stereo and multi-channel recordings tell where a sound comes from.** For
  every channel, the level difference to the first channel (dB) and the
  arrival delay (in samples, by GCC-PHAT) of the marked part are measured.
  Overtones and resonances whose channel ratio or delay does not fit come
  from another source and are left out, and a repetition of the call from a
  different place (the neighbour singing the same song) is not taken for
  another occurrence. How strict that check is follows from the marked part
  itself: in a room, reflections make every bin see a slightly different
  level ratio and phase, so the tolerances grow with how much the marked
  part's own bins scatter, and a narrow band that pins the delay down only
  loosely counts every delay it cannot tell apart as a match. Mono
  recordings work exactly as before.
* **Find out what a recording is made of.** *Find sounds* learns the
  recurring spectral patterns of the recording (or of the selected stretch)
  by non-negative matrix factorisation: each pattern has a fixed spectrum
  and only its loudness changes over time, and the background noise gets a
  pattern of its own so the others describe sounds. Patterns whose loudness
  rises and falls together and that, in stereo, come from the same place
  are suggested as one sound. The suggestions appear in the filter list
  switched off; pick one to see it in red, switch it on to suppress or
  extract it, and keep, rename, merge, split or discard it. Each pattern is
  a part that can be switched off. You choose how many patterns to look
  for; a handful of distinct sounds works best.
* **Background noise removal that keeps the sound intact.** Instead of a gate
  that deletes everything below the noise floor (which eats attack and decay
  and changes the balance of partials), the noise filter uses a soft spectral
  gain with decision-directed SNR smoothing (Wiener or MMSE log-spectral
  amplitude). Loud sounds pass almost unchanged, and the gain never drops
  below a configurable floor. The noise spectrum is estimated adaptively or
  learned from a noise-only stretch you select.
* **A filter building kit.** Filters are listed on the right; each can be
  named, switched on and off, shown or hidden in the spectrogram, set to
  *suppress* or *extract*, and its parts (marked area, harmonics,
  co-modulated resonances, the same sound elsewhere) can be disabled
  temporarily.
* **Responsive with long recordings.** Processing runs on background threads
  in chunks of a few seconds, nearest to the play head, the selection and the
  visible area first. A one-hour recording is usable immediately; the rest is
  processed while you work.

## Download

Each [release](https://github.com/gunterkoenigsmann/noiseextractor/releases)
has a Debian package and an AppImage for Linux, a zip and an installer for
Windows and a disk image for macOS, each for Intel/AMD and Arm processors.
They hold the GUI and the command-line tool. The packages are not signed
yet; [docs/release.md](docs/release.md) says how to get past the warnings,
and how a release is made.

On Linux there is also a snap (`sudo snap install noiseextractor`, or
`--edge` for the newest development build; the command-line tool is then
`noiseextractor.cli`). It is rebuilt every week so that it picks up Ubuntu's
security updates; [docs/snap.md](docs/snap.md) has the details.

## Building

You need a stable Rust toolchain, CMake and a C++ compiler. The GUI uses
[wxDragon](https://github.com/AllenDang/wxDragon), which downloads and compiles
wxWidgets on the first build (that takes a few minutes).

* **Linux** (Debian/Ubuntu):
  `sudo apt-get install libgtk-3-dev libasound2-dev libpng-dev libjpeg-dev libtiff-dev libexpat1-dev libclang-dev pkg-config cmake`
* **macOS**: Xcode command line tools and CMake.
* **Windows**: Visual Studio (C++ build tools) and CMake.

```sh
cargo run --release -p noiseextractor -- recording.flac    # the GUI
cargo run --release -p noiseextractor-cli -- --help        # the command-line tool
cargo test --workspace                                     # all tests
```

`cargo build`/`cargo test` at the workspace root without `--workspace` only
build the headless crates (core and CLI), which is quick and needs no GUI
libraries.

## Using the GUI

| Action | How |
| --- | --- |
| Open a recording | File > Open audio (Ctrl+O), or pass it on the command line |
| Mark a sound | Drag a rectangle in the spectrogram |
| Turn it into a filter | *Suppress selection* or *Extract selection* |
| Switch filters on / off | Click a filter's box, or press Space to switch all picked filters |
| See what a filter does | Pick it in the filter list: its target is drawn (green / red / orange), for a noise reduction what it takes away (red); *Show in spectrogram* keeps it drawn when another filter is picked |
| Tune what belongs to it and what counts as a repetition | Sliders in the *Detection settings* pane (they edit the fresh selection, or the selected sound filter; hover a slider for an explanation) |
| Set the play head | Click |
| Play / pause | Space (plays the selection if there is one), or Ctrl+P |
| Hear only the marked sound / everything but it | View > *Hear the picked sound only* (Ctrl+2) / *Hear everything but the picked sound* (Ctrl+3); *Hear everything* (Ctrl+1) goes back. The sound is the selection being edited, else the filter(s) picked in the list (Ctrl/Shift+click picks several, which are then heard or removed together) (whether it suppresses or extracts, and even if it is switched off or a suggestion). Other sounds are suppressed in solo, so where they overlap only the picked sound's share is heard. Blue shows what you hear; export is not affected |
| Scrub (hear the audio under the pointer) | Drag with the right mouse button |
| Zoom time / scroll / zoom frequency | Wheel / Shift+wheel / Ctrl+wheel |
| Jump elsewhere in a long recording | Drag the frame in the waveform overview, or click beside it |
| See where else in the recording a sound was found | Mark it (or pick its filter): the waveform overview draws in orange how loud what the filter finds is at each time; it fills in as the recording is processed |
| Show / hide panes (filters, detection settings, overview, log) | View menu, or the pane's close button; panes can be dragged to other edges or floated, and the layout is remembered |
| Find the sounds a recording is made of | *Find sounds* (Ctrl+F); how many patterns to look for and whether they are grouped is set in File > Settings > Find sounds (only the selected stretch is used to learn, if there is a selection); *Keep*, *Merge with...*, *Split* and *Discard suggestions* edit the result |
| Learn the background noise | Mark a noise-only stretch, *Learn noise from selection* |
| Save the filter list | File > Save project (Ctrl+S). The project remembers its recording (relative to the project file when both live in one folder tree, so the folder can be moved), and opening the project opens that recording again; if it is gone, the filters apply to the recording that is open |
| Reopen a recent recording or project | File > Recent recordings / Recent projects (nine of each are remembered; a file that is gone is dropped from the list) |
| Write the result | File > Export processed audio (Ctrl+E); the file type list (or the extension you type) picks WAV, FLAC, Ogg Vorbis, MP3, Opus or Apple Lossless, see [Export formats](#export-formats) |
| Make a quiet extracted sound louder in the export | File > Settings > Export > *Normalize the exported audio*: the whole file is scaled by one gain so its loudest sample reaches the chosen level (default -1 dBFS); the levels inside it keep their ratios |
| Make the spectrogram brighter / show quieter detail | File > Settings > Spectrogram (the levels drawn black and at full colour; changes show at once) |
| Look up what the colours mean | Help > Colours and mouse |

File > Settings (Ctrl+,) holds what is set once and rarely changed, on
pages listed down its left side like losslessaudiocut's; it is remembered
between runs.

## Command-line tool

`noiseextractor-cli` runs the same processing without a window, e.g. for
batch jobs or tests:

```sh
noiseextractor-cli denoise in.flac -o clean.wav --noise-from 0.0:1.5 --save-project p.json
noiseextractor-cli mark in.flac --time 12.3:13.1 --freq 2000:4500 --name blackbird \
    --extract --project p.json --save-project p.json
noiseextractor-cli discover in.flac --sounds 6 --enable 2 --extract --save-project p.json
noiseextractor-cli process in.flac --project p.json -o out.wav
noiseextractor-cli process in.flac --project p.json --solo 2,3 -o two-sounds.wav   # or --without 2,3
noiseextractor-cli process in.flac --project p.json -o out.opus --bitrate 96   # see Export formats
noiseextractor-cli process in.flac --project p.json --solo 2 --normalize -o loud.wav  # peak at -1 dBFS (or --normalize=-3)
noiseextractor-cli spectrogram in.flac --project p.json -o view.png --max-hz 10000
```

`mark` prints which harmonic ratios it found, where the marked sound comes
from (level and delay per channel, for multi-channel files) and where else in
the recording the same sound occurs, so you can check what a filter will grab before
applying it. `discover` prints the sounds it found, numbered, with the
patterns they consist of (tonal or noise-like, frequency band, direction);
`--enable` switches the given ones on.

## Export formats

The output file's extension picks the format (`--format` overrides it on the
command line). All encoders are compiled into the program, so every format
works the same on Linux, Windows and macOS.

| Format | Extension | What you get |
| --- | --- | --- |
| WAV | `.wav` | 32-bit float, exactly what the filters produced |
| FLAC | `.flac` | 24-bit, lossless |
| Ogg Vorbis | `.ogg` | quality 6 (≈190 kbit/s stereo), or `--bitrate` |
| MP3 | `.mp3` | LAME VBR `-V2` (≈190 kbit/s stereo), or constant `--bitrate`; mono or stereo; rates above 48 kHz are resampled |
| Opus | `.opus` | 64 kbit/s per channel, or `--bitrate`; always 48 kHz (resampled); mono or stereo |
| Apple Lossless | `.m4a` | 16-bit with dither, lossless from there on; plays in iTunes/Music and on iPhones |

AAC is missing on purpose: AAC encoders are still licensed through a patent
pool, and the one good free encoder (FDK AAC) has a licence that is not
compatible with the GPL. Apple Lossless, by contrast, has been open source
(Apache 2.0) and royalty free since 2011. Exported Opus files cannot be
opened again yet (the decoder library has no Opus support).

## How filters combine

Every filter describes its target as a soft mask m between 0 and 1 per
time-frequency bin. *Suppress* multiplies the signal by √(1 − m²) (removes the
target's share of the energy), *extract* keeps m. All active extract filters
are combined as a union, then every suppress filter is applied. Suppressing and
extracting the same target therefore add up (in power) to the original.

Where a sound you extract and a sound you suppress claim the same bins (a
bird's overtone crossing a whistle), the bin is shared out instead of being
judged by each mask alone: every marked sound knows how loud its partials and
co-modulated bands are relative to its fundamental, so its loudness in each
frame is measured on the bins nobody else claims, and its expected power in
the shared bins follows from that. The extracted sound keeps its share, the
suppressed one loses its share. In stereo recordings, how well a shared bin's
level ratio and phase between the channels fit each sound's direction weighs
in as well. Automatically found sounds take part in this too: their patterns
predict their power in every bin directly. In a test where a whistle crosses a kept call's overtone, the
overtone now keeps its level (−0.3 dB) instead of losing 28 dB.

## Project layout

| Crate | Purpose |
| --- | --- |
| `crates/core` (`noiseextractor-core`) | STFT, noise reduction, selection analysis, filter model, background engine, renderer. No GUI dependencies. |
| `crates/cli` (`noiseextractor-cli`) | Command-line front end and end-to-end tests. |
| `crates/gui` (`noiseextractor`) | wxDragon GUI and audio playback (cpal). |

## Planned

* Tolerate repetitions that are faster/slower or transposed by more than a
  few bins (time-stretch and pitch-scale invariant matching).
* Smoother adaptive noise estimate across chunk borders.
* Undo/redo and editing a filter's rectangle after creating it.

## License

GPL-3.0-or-later, see [LICENSE](LICENSE).
