# noiseextractor

**Noise reduction and sound extraction for audio recordings**, from Python
or the command line:

- **Remove background noise** (hiss, hum, wind, traffic rumble, room
  noise) from a recording: `denoise()`.
- **Remove one particular sound** (a barking dog, a cricket, a car, a
  cough) or **extract it** (a bird song, an owl call, a voice) by marking a
  small part of it: `mark()` and `process()`.
- **Find the sounds a recording is made of** automatically and keep or
  drop each: `discover()`.

It works with soft spectral masks instead of a noise gate: what remains
keeps its attack, decay and the loudness ratios of its overtones, so quiet
sounds don't vanish and loud ones don't get "pumping" artefacts. Marking
part of a sound is enough; the analysis follows its pitch, finds its
overtones and resonances, the frequencies that rise and fall with it, its
direction in stereo recordings, and every other place in the recording
where the same sound occurs.

This is the engine of the [noiseextractor](https://github.com/gunterkoenigsmann/noiseextractor)
desktop application (Linux, Windows, macOS), where the same filters are
built interactively on a spectrogram. Projects (`.nxp`) are shared between
the application, this package and the command line tool.

```sh
pip install noiseextractor            # NumPy support: pip install "noiseextractor[numpy]"
```

Wheels for Linux (x86_64, aarch64), macOS (Apple silicon, Intel) and
Windows (x64) for every CPython from 3.10 on; no system libraries needed.
Reads WAV, FLAC, Ogg Vorbis, MP3, AAC/M4A and ALAC; writes WAV, FLAC, Ogg
Vorbis, MP3, Opus and ALAC (`.m4a`).

## Remove background noise

```python
import noiseextractor as nx

nx.denoise("field-recording.flac", "clean.flac")

# Stronger, with the noise learned from a noise-only stretch (0.0 .. 1.5 s),
# and the result normalised to -1 dBFS:
nx.denoise("interview.mp3", "clean.mp3", reduction_db=24,
           noise_from=(0.0, 1.5), normalize=-1.0)
```

## Remove or extract one sound

Mark a short part of the sound, a rectangle in the spectrogram: when it is
and which frequencies it covers. One call of a bird is enough to find the
others.

```python
report = nx.mark("garden.flac", time=(12.3, 13.1), freq=(2000, 4500),
                 name="blackbird", extract=True, save_project="garden.nxp")
print(report)                       # pitch, overtones, where it occurs again
for o in report.occurrences:
    print(f"{o.time_s:.2f} s, match {o.score:.2f}")

# Filter 1 of a new project is the background noise reduction, so the
# marked sound is filter 2.
nx.process("garden.flac", "blackbird-only.wav", project="garden.nxp", solo=[2])
nx.process("garden.flac", "without-blackbird.wav", project="garden.nxp", without=[2])
```

To suppress a sound (a dog, a cough), mark it without `extract=True` and
run `nx.process("in.flac", "out.flac", project="garden.nxp")`.

A sound that is not a rectangle in the spectrogram (a U-shaped sweep, a
slanted chirp, a call right next to another one) can be marked by the
corners of a polygon around it instead, as `(seconds, Hz)` pairs; only
what lies inside counts as marked:

```python
nx.mark("garden.flac", outline=[(1.0, 600), (1.5, 250), (2.0, 600),
                                (2.0, 700), (1.5, 350), (1.0, 700)],
        name="swoop", project="garden.nxp", save_project="garden.nxp")
```

## Find the sounds automatically

```python
for s in nx.discover("night.flac", sounds=6):
    print(s)                        # "suggestion 1 (42 % of the sound): ..."

# Keep suggestion 2 and extract it:
nx.discover("night.flac", sounds=6, enable=[2], extract=True,
            save_project="night.nxp")
nx.process("night.flac", "owl.wav", project="night.nxp")
```

## Samples in memory

The `*_array` functions take and return float samples in -1..1: a NumPy
array of shape `(frames,)` or `(channels, frames)`, or plain sequences
(then `array.array("f")` comes back).

```python
samples, rate = nx.load("in.wav")          # (channels, frames) float32
clean = nx.denoise_array(samples, rate)
birds = nx.process_array(samples, rate, project="garden.nxp", solo=[2])
```

## Spectrogram

```python
nx.spectrogram("garden.flac", "garden.png", project="garden.nxp", log=True)
```

Blue is what remains, green what a filter's marked part covers, red what
the analysis found to belong to it, orange both.

## Command line

The package installs `noiseextractor-cli` (also `python -m noiseextractor`):

```sh
noiseextractor-cli denoise in.flac -o clean.flac
noiseextractor-cli mark in.flac --time 12.3:13.1 --freq 2000:4500 \
    --name blackbird --extract --save-project birds.nxp
noiseextractor-cli process in.flac --project birds.nxp --solo 2 -o birds.wav
noiseextractor-cli discover in.flac --sounds 6
noiseextractor-cli --help
```

## API

| Function | What |
|---|---|
| `denoise(input, output, *, reduction_db=18, rule="lsa", noise_from=None, normalize=None, format=None, bitrate=None, save_project=None)` | Background noise reduction. |
| `mark(input, *, time, freq, name, extract=False, project=None, save_project=None)` | Mark part of a sound; returns a `MarkReport` and adds a filter to the project. Instead of `time` and `freq`, `outline=[(seconds, Hz), ...]` marks a free-form region. |
| `discover(input, *, sounds=6, time=None, grouping=True, enable=None, extract=False, project=None, save_project=None)` | Suggest a filter per sound found; returns `Suggestion`s. |
| `process(input, output, *, project=None, solo=None, without=None, normalize=None, format=None, bitrate=None)` | Apply a project; `solo`/`without` keep or drop filters' sounds (numbered from 1). |
| `denoise_array(samples, rate, ...)`, `process_array(samples, rate, ...)` | The same on samples in memory. |
| `load(path)` | Decode a file into `(samples, sample_rate)`. |
| `spectrogram(input, output, *, project=None, width=1200, height=400, max_hz=None, log=False)` | PNG of the processed spectrogram. |
| `new_project(path)` | A project with background noise reduction only. |

Times are seconds, frequencies Hz. File errors raise `noiseextractor.Error`,
bad arguments `ValueError`. Every function releases the interpreter lock
while it works and takes `threads=` (default: one per CPU core).

## License

LGPL-3.0-or-later: you may use the package from programs under any
license, open or closed; changes to noiseextractor itself must be shared
under the same license. (The desktop application is GPL-3.0-or-later.)
Source, issues and the desktop application:
<https://github.com/gunterkoenigsmann/noiseextractor>.
