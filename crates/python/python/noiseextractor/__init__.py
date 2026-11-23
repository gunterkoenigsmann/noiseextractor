"""Remove background noise from audio recordings, or suppress or extract
single sounds (a bird song, a cricket, traffic, a hum) with soft spectral
masks that keep the attack, decay and timbre of what remains.

The same code as the noiseextractor desktop application and its command
line tool (``noiseextractor-cli``, installed with this package):

>>> import noiseextractor as nx
>>> nx.denoise("field.flac", "clean.flac")                 # background noise
>>> r = nx.mark("field.flac", time=(12.3, 13.1), freq=(2000, 4500),
...             name="blackbird", extract=True, save_project="birds.nxp")
>>> [o.time_s for o in r.occurrences]                      # same call elsewhere
>>> nx.process("field.flac", "blackbird.wav", project="birds.nxp", solo=[2])

Arrays work too (NumPy if installed, otherwise ``array.array``):

>>> clean = nx.denoise_array(samples, 48000)

Times are seconds, frequencies Hz. Errors reading or writing files raise
:class:`Error`; bad arguments raise :class:`ValueError`.
"""

from __future__ import annotations

import array
import numbers
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

from . import _native

__version__: str = _native.__version__

#: Reading, processing or writing a file failed.
Error = _native.Error

#: A file path.
PathLike = Union[str, "os.PathLike[str]"]

__all__ = [
    "Error",
    "ExportResult",
    "Partial",
    "Direction",
    "Occurrence",
    "MarkReport",
    "Pattern",
    "Suggestion",
    "denoise",
    "process",
    "mark",
    "discover",
    "spectrogram",
    "new_project",
    "load",
    "denoise_array",
    "process_array",
    "__version__",
]


@dataclass(frozen=True)
class ExportResult:
    """What writing a processed file did."""

    #: Loudest sample of the processed audio before normalisation (1.0 =
    #: full scale).
    peak: float
    #: Gain applied by ``normalize`` in dB (0.0 without normalisation).
    gain_db: float


@dataclass(frozen=True)
class Partial:
    """An overtone or resonance found to belong to a marked sound."""

    #: Frequency as a multiple of the sound's pitch (2.0 = octave).
    ratio: float
    #: How sure the analysis is that it belongs to the sound (0..1).
    score: float


@dataclass(frozen=True)
class Direction:
    """Where a sound comes from: one channel relative to the first."""

    #: The other channel (1 = second channel).
    channel: int
    #: Its level relative to channel 0 in dB.
    level_db: float
    #: Its delay relative to channel 0 in samples.
    delay_samples: float


@dataclass(frozen=True)
class Occurrence:
    """A place where a marked sound occurs (again)."""

    #: Start in seconds.
    time_s: float
    #: Shape match with the marked sound (0..1).
    score: float
    #: Frequency shift against the marked sound in STFT bins.
    shift_bins: int
    #: Length in seconds for pulsed sounds, else ``None`` (as long as the
    #: marked part).
    length_s: Optional[float]
    #: Agreement of the direction (1 = same place), ``None`` for mono.
    same_place: Optional[float]
    #: Whether this is the marked sound itself.
    marked: bool


@dataclass(frozen=True)
class MarkReport:
    """What :func:`mark` found out about the marked sound."""

    #: The filter's name.
    name: str
    #: ``True`` if the filter extracts the sound, ``False`` if it
    #: suppresses it.
    extract: bool
    #: Band ``(low, high)`` in Hz of a noise-like sound without pitch.
    broadband_hz: Optional[Tuple[float, float]]
    #: Lowest and highest dominant frequency in Hz of a pitched sound.
    pitch_hz: Optional[Tuple[float, float]]
    #: Overtones and resonances that belong to the sound.
    partials: Tuple[Partial, ...]
    #: Number of frequency bins whose level follows the sound.
    comodulated_bins: int
    #: Stereo cues; empty for mono recordings.
    directions: Tuple[Direction, ...]
    #: Pulse rate in Hz for pulsed sounds (crickets, woodpeckers).
    pulse_hz: Optional[float]
    #: Where the same sound occurs in the recording, ``None`` if it has
    #: nothing distinctive to search for.
    occurrences: Optional[Tuple[Occurrence, ...]]
    #: The report as the command line tool prints it.
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Pattern:
    """One spectral pattern learned by :func:`discover`."""

    #: Its number (1 = first learned).
    number: int
    #: A short description, such as ``"tonal, peak 660 Hz"``.
    label: str
    #: ``(level dB, delay samples)`` of the second channel, if stereo.
    direction: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class Suggestion:
    """A sound :func:`discover` suggests as a filter."""

    #: The number to pass in ``enable`` (from 1).
    number: int
    #: Share of the recording's sound it explains (0..1).
    share: float
    #: The patterns it is made of.
    patterns: Tuple[Pattern, ...]
    #: The suggestion as the command line tool prints it.
    text: str

    def __str__(self) -> str:
        return self.text


def _path(p: Optional[PathLike]) -> Optional[str]:
    """``os.fspath`` that lets ``None`` through."""
    return None if p is None else os.fspath(p)


def _range(r: Optional[Sequence[float]], what: str) -> Optional[Tuple[float, float]]:
    """A ``(start, end)`` pair of floats, checked."""
    if r is None:
        return None
    try:
        a, b = r
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a (start, end) pair, not {r!r}") from None
    return float(a), float(b)


def _numbers(ns: Optional[Iterable[int]]) -> list:
    """A list of filter numbers (from 1); a single int is accepted too."""
    if ns is None:
        return []
    if isinstance(ns, int):
        return [ns]
    return [int(n) for n in ns]


def _export_result(d: Optional[dict]) -> Optional[ExportResult]:
    return None if d is None else ExportResult(peak=d["peak"], gain_db=d["gain_db"])


def denoise(
    input: PathLike,
    output: Optional[PathLike] = None,
    *,
    reduction_db: float = 18.0,
    rule: str = "lsa",
    noise_from: Optional[Tuple[float, float]] = None,
    normalize: Optional[float] = None,
    format: Optional[str] = None,
    bitrate: Optional[int] = None,
    save_project: Optional[PathLike] = None,
    threads: Optional[int] = None,
) -> Optional[ExportResult]:
    """Reduces the steady background noise (hiss, hum, wind, traffic rumble)
    of a recording and writes the result to ``output``.

    Uses soft gains (MMSE log-spectral amplitude by default, ``rule="wiener"``
    for a Wiener gain) against an adaptive noise estimate, so loud sounds
    keep their attack, decay and the ratios of their partials.

    Args:
        input: Audio file (WAV, FLAC, Ogg Vorbis, MP3, AAC/M4A, ALAC, ...).
        output: File to write; its extension picks the format (.wav .flac
            .ogg .mp3 .opus .m4a) unless ``format`` is given. ``None`` only
            saves the project.
        reduction_db: Maximum attenuation in dB.
        rule: ``"lsa"`` or ``"wiener"``.
        noise_from: Learn the noise from this noise-only stretch
            ``(start, end)`` in seconds instead of estimating it adaptively.
        normalize: Scale the output so its loudest sample reaches this many
            dBFS (for example ``-1.0``); one gain for the whole file.
        format: ``"wav"``, ``"flac"``, ``"vorbis"``, ``"mp3"``, ``"opus"`` or
            ``"alac"``.
        bitrate: kbit/s for the lossy formats (default: transparent).
        save_project: Also save the project (``.nxp``) for the GUI.
        threads: Worker threads (default: one per CPU core).
    """
    return _export_result(
        _native.denoise(
            os.fspath(input),
            _path(output),
            float(reduction_db),
            rule,
            _range(noise_from, "noise_from"),
            None if normalize is None else float(normalize),
            format,
            bitrate,
            _path(save_project),
            threads,
        )
    )


def process(
    input: PathLike,
    output: PathLike,
    *,
    project: Optional[PathLike] = None,
    solo: Optional[Iterable[int]] = None,
    without: Optional[Iterable[int]] = None,
    normalize: Optional[float] = None,
    format: Optional[str] = None,
    bitrate: Optional[int] = None,
    threads: Optional[int] = None,
) -> ExportResult:
    """Applies the enabled filters of a project (saved by the GUI,
    :func:`mark`, :func:`discover` or :func:`denoise`) and writes the result.

    Args:
        input: Audio file.
        output: File to write; the extension picks the format.
        project: Project file (``.nxp``); default: background noise
            reduction only.
        solo: Keep only the sounds of these filters (numbered from 1), for
            example to extract a bird song.
        without: Keep everything but the sounds of these filters.
        normalize, format, bitrate, threads: As for :func:`denoise`.
    """
    return _export_result(
        _native.process(
            os.fspath(input),
            os.fspath(output),
            _path(project),
            _numbers(solo),
            _numbers(without),
            None if normalize is None else float(normalize),
            format,
            bitrate,
            threads,
        )
    )


def _mark_report(d: dict) -> MarkReport:
    occ = d["occurrences"]
    return MarkReport(
        name=d["name"],
        extract=d["extract"],
        broadband_hz=d["broadband_hz"],
        pitch_hz=d["pitch_hz"],
        partials=tuple(Partial(*p) for p in d["partials"]),
        comodulated_bins=d["comodulated_bins"],
        directions=tuple(Direction(*x) for x in d["directions"]),
        pulse_hz=d["pulse_hz"],
        occurrences=None if occ is None else tuple(Occurrence(**o) for o in occ),
        text=d["text"],
    )


def mark(
    input: PathLike,
    *,
    time: Tuple[float, float],
    freq: Tuple[float, float],
    name: str = "Marked noise",
    extract: bool = False,
    project: Optional[PathLike] = None,
    save_project: Optional[PathLike] = None,
) -> MarkReport:
    """Marks a part of one sound, a rectangle in the spectrogram, and turns
    it into a filter for the whole sound.

    The analysis follows the marked part's pitch, finds its overtones and
    resonances and the frequencies that rise and fall with it, takes its
    direction in stereo recordings, and searches the whole recording for
    the same sound. The filter is added to ``project`` (or a new project
    with background noise reduction) and saved to ``save_project``; run it
    with :func:`process`.

    Args:
        input: Audio file.
        time: ``(start, end)`` of the marked part in seconds.
        freq: ``(low, high)`` of the marked part in Hz.
        name: Name of the new filter.
        extract: Extract the sound (keep only it with ``process(...,
            solo=[n])``) instead of suppressing it.
        project: Project to add the filter to.
        save_project: Where to save the extended project.

    Returns:
        What the analysis found; ``str(report)`` is a readable summary.
    """
    t = _range(time, "time")
    f = _range(freq, "freq")
    assert t is not None and f is not None
    return _mark_report(
        _native.mark(os.fspath(input), t, f, name, bool(extract), _path(project), _path(save_project))
    )


def discover(
    input: PathLike,
    *,
    sounds: int = 6,
    time: Optional[Tuple[float, float]] = None,
    grouping: bool = True,
    enable: Optional[Iterable[int]] = None,
    extract: bool = False,
    project: Optional[PathLike] = None,
    save_project: Optional[PathLike] = None,
) -> Tuple[Suggestion, ...]:
    """Finds the sounds a recording is made of (non-negative matrix
    factorisation against the background noise) and suggests a filter for
    each, without anything to mark by hand.

    Suggestions are added to ``project`` switched off, except those numbered
    in ``enable``, and the project is saved to ``save_project``.

    Args:
        input: Audio file.
        sounds: Number of spectral patterns to look for.
        time: Learn only from this ``(start, end)`` in seconds.
        grouping: Group patterns that rise and fall together (and come from
            the same direction) into one suggestion.
        enable: Suggestions to switch on and keep (numbered from 1).
        extract: Switched-on suggestions extract instead of suppress.
        project: Project to add the suggestions to.
        save_project: Where to save the extended project.
    """
    found = _native.discover(
        os.fspath(input),
        int(sounds),
        _range(time, "time"),
        bool(grouping),
        _numbers(enable),
        bool(extract),
        _path(project),
        _path(save_project),
    )
    return tuple(
        Suggestion(
            number=s["number"],
            share=s["share"],
            patterns=tuple(Pattern(**p) for p in s["patterns"]),
            text=s["text"],
        )
        for s in found
    )


def spectrogram(
    input: PathLike,
    output: PathLike,
    *,
    project: Optional[PathLike] = None,
    width: int = 1200,
    height: int = 400,
    max_hz: Optional[float] = None,
    log: bool = False,
    threads: Optional[int] = None,
) -> None:
    """Renders the spectrogram as a PNG with the GUI's colours: blue is
    what remains, green what a filter's marked part covers, red what the
    analysis found to belong to it, orange both.

    Args:
        input: Audio file.
        output: PNG file to write.
        project: Project whose filters to apply (default: none).
        width, height: Image size in pixels.
        max_hz: Highest frequency shown (default: half the sample rate).
        log: Logarithmic frequency axis.
        threads: Worker threads.
    """
    _native.spectrogram(
        os.fspath(input),
        os.fspath(output),
        _path(project),
        int(width),
        int(height),
        None if max_hz is None else float(max_hz),
        bool(log),
        threads,
    )


def new_project(path: PathLike) -> None:
    """Writes a project with the default filters (background noise
    reduction), a starting point for :func:`mark` and :func:`process`."""
    _native.new_project(os.fspath(path))


# ---------------------------------------------------------------------------
# Samples in memory
# ---------------------------------------------------------------------------


def _numpy() -> Any:
    """NumPy if it is installed (it is optional), else ``None``."""
    return sys.modules.get("numpy") or _try_import_numpy()


def _try_import_numpy() -> Any:
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _to_channels(samples: Any) -> Tuple[list, bool]:
    """Float32 ``bytes`` per channel, and whether the input was mono (one
    dimension), so the result can have the same shape."""
    np = _numpy()
    if np is not None and isinstance(samples, np.ndarray):
        a = np.asarray(samples, dtype=np.float32)
        if a.ndim == 1:
            return [a.tobytes()], True
        if a.ndim == 2:
            return [np.ascontiguousarray(c).tobytes() for c in a], False
        raise ValueError(f"samples must have 1 or 2 dimensions (channels, frames), not {a.ndim}")
    seq = list(samples)
    if seq and not isinstance(seq[0], numbers.Real):
        # A sequence of channels.
        return [array.array("f", c).tobytes() for c in seq], False
    return [array.array("f", seq).tobytes()], True


def _from_channels(chans: list, mono: bool, like: Any) -> Any:
    """The processed channels in the input's shape and kind."""
    np = _numpy()
    if np is not None and isinstance(like, np.ndarray):
        out = np.stack([np.frombuffer(c, dtype=np.float32) for c in chans])
        return out[0] if mono else out
    arrays = []
    for c in chans:
        a = array.array("f")
        a.frombytes(c)
        arrays.append(a)
    return arrays[0] if mono else arrays


def load(path: PathLike) -> Tuple[Any, int]:
    """Decodes an audio file into ``(samples, sample_rate)``: a float32
    NumPy array of shape ``(channels, frames)`` if NumPy is installed, else
    a list of ``array.array("f")``, one per channel."""
    chans, rate = _native.load(os.fspath(path))
    np = _numpy()
    if np is not None:
        return np.stack([np.frombuffer(c, dtype=np.float32) for c in chans]), rate
    return _from_channels(chans, False, None), rate


def denoise_array(
    samples: Any,
    sample_rate: int,
    *,
    reduction_db: float = 18.0,
    rule: str = "lsa",
    noise_from: Optional[Tuple[float, float]] = None,
    threads: Optional[int] = None,
) -> Any:
    """:func:`denoise` for samples in memory.

    Args:
        samples: Floats in -1..1: a NumPy array of shape ``(frames,)`` or
            ``(channels, frames)``, or a sequence of floats, or a sequence
            of channels.
        sample_rate: Samples per second.
        reduction_db, rule, noise_from, threads: As for :func:`denoise`.

    Returns:
        The processed samples, same shape and kind as ``samples``
        (``array.array("f")`` for plain sequences).
    """
    chans, mono = _to_channels(samples)
    out = _native.denoise_samples(
        chans, int(sample_rate), float(reduction_db), rule, _range(noise_from, "noise_from"), threads
    )
    return _from_channels(out, mono, samples)


def process_array(
    samples: Any,
    sample_rate: int,
    *,
    project: Optional[PathLike] = None,
    solo: Optional[Iterable[int]] = None,
    without: Optional[Iterable[int]] = None,
    threads: Optional[int] = None,
) -> Any:
    """:func:`process` for samples in memory (see :func:`denoise_array`
    for ``samples`` and the result)."""
    chans, mono = _to_channels(samples)
    out = _native.process_samples(
        chans, int(sample_rate), _path(project), _numbers(solo), _numbers(without), threads
    )
    return _from_channels(out, mono, samples)
