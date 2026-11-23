"""Tests of the Python package, run against the installed wheel:

    pip install noiseextractor-*.whl pytest && pytest crates/python/tests

They check the Python layer (argument conversion, result classes, errors,
the ``noiseextractor-cli`` command, arrays with and without NumPy) on
synthetic recordings, reading the results with an independent WAV reader
(``read_wav`` below, not the package's decoder) where they are files. The
signal processing itself is tested in depth by the Rust crates.
"""

from __future__ import annotations

import array
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

import noiseextractor as nx

RATE = 16000


def noisy_tone(seconds: float = 3.0, tone: tuple = (1.0, 2.0)) -> list:
    """Low noise everywhere, a 660 Hz tone with an octave overtone during
    ``tone`` (seconds): the CLI tests' recording."""
    state = 5
    out = []
    for i in range(int(seconds * RATE)):
        # A 64-bit LCG, so the noise is the same on every platform.
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        n = ((state >> 40) / (1 << 24) - 0.5) * 0.04
        t = i / RATE
        s = 0.0
        if tone[0] <= t < tone[1]:
            w = 2 * math.pi * 660 * t
            s = 0.4 * math.sin(w) + 0.2 * math.sin(2 * w)
        out.append(s + n)
    return out


def write_wav16(path: Path, samples: list, rate: int = RATE) -> None:
    """A 16-bit mono WAV, written by the standard library."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        ints = [max(-32768, min(32767, round(s * 32767))) for s in samples]
        w.writeframes(struct.pack(f"<{len(ints)}h", *ints))


def read_wav(path: Path) -> tuple:
    """``(channels, rate)`` of a 16-bit PCM or 32-bit float WAV, read
    independently of the package (the standard library's ``wave`` cannot
    read float WAVs, which the package writes)."""
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    pos, fmt, body = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        chunk = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = chunk
        elif cid == b"data":
            body = chunk
        pos += 8 + size + (size & 1)
    assert fmt is not None and body is not None
    tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == 0xFFFE:
        # WAVE_FORMAT_EXTENSIBLE: the sub-format GUID starts with the tag.
        tag = struct.unpack("<H", fmt[24:26])[0]
    if tag == 3 and bits == 32:
        values = struct.unpack(f"<{len(body) // 4}f", body)
    elif tag == 1 and bits == 16:
        values = [v / 32768 for v in struct.unpack(f"<{len(body) // 2}h", body)]
    else:
        raise AssertionError(f"unexpected WAV format {tag}/{bits}")
    return [list(values[c::channels]) for c in range(channels)], rate


def rms(x) -> float:
    return math.sqrt(sum(v * v for v in x) / len(x))


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> tuple:
    """A noisy tone as a 16-bit WAV and its samples."""
    samples = noisy_tone()
    path = tmp_path_factory.mktemp("rec") / "in.wav"
    write_wav16(path, samples)
    return path, samples


def test_version() -> None:
    assert nx.__version__[0].isdigit()


def test_denoise_file_reduces_noise_and_keeps_the_tone(recording, tmp_path) -> None:
    path, samples = recording
    out = tmp_path / "clean.wav"
    result = nx.denoise(path, out)
    assert isinstance(result, nx.ExportResult)
    assert result.gain_db == 0.0
    (clean,), rate = read_wav(out)
    assert rate == RATE and len(clean) == len(samples)
    assert rms(clean[3000:12000]) < 0.3 * rms(samples[3000:12000]), "noise not reduced"
    assert rms(clean[20000:28000]) > 0.9 * rms(samples[20000:28000]), "tone damaged"


def test_denoise_normalizes_and_saves_a_project(recording, tmp_path) -> None:
    path, _ = recording
    out = tmp_path / "loud.wav"
    project = tmp_path / "p.nxp"
    result = nx.denoise(path, out, normalize=-1.0, save_project=project, rule="wiener", noise_from=(0, 0.9))
    (clean,), _ = read_wav(out)
    peak = max(abs(v) for v in clean)
    assert abs(20 * math.log10(peak) + 1.0) < 0.05
    assert result.gain_db > 0
    assert project.exists()
    # Without an output only the project is written.
    assert nx.denoise(path, save_project=tmp_path / "q.nxp") is None
    assert (tmp_path / "q.nxp").exists()


def test_mark_finds_the_overtone_and_extracting_keeps_only_the_tone(recording, tmp_path) -> None:
    path, samples = recording
    project = tmp_path / "tone.nxp"
    report = nx.mark(path, time=(1.2, 1.6), freq=(500, 800), name="tone", extract=True, save_project=project)
    assert isinstance(report, nx.MarkReport)
    assert report.name == "tone" and report.extract
    lo, hi = report.pitch_hz
    assert 600 < lo <= hi < 720
    assert any(abs(p.ratio - 2.0) < 0.05 for p in report.partials), report.text
    assert "filter \"tone\"" in str(report)
    out = tmp_path / "tone.wav"
    # Filter 1 is the background noise reduction, 2 the marked tone.
    nx.process(path, out, project=project, solo=[2])
    (tone,), _ = read_wav(out)
    assert rms(tone[3000:12000]) < 0.2 * rms(samples[3000:12000]), "noise kept"
    assert rms(tone[20000:28000]) > 0.7 * rms(samples[20000:28000]), "tone lost"


def test_discover_suggests_and_enables(recording, tmp_path) -> None:
    path, _ = recording
    project = tmp_path / "found.nxp"
    found = nx.discover(path, sounds=3, enable=[1], save_project=project)
    assert found and all(isinstance(s, nx.Suggestion) for s in found)
    assert [s.number for s in found] == list(range(1, len(found) + 1))
    assert all(0 <= s.share <= 1 and s.patterns for s in found)
    assert "suggestion 1" in str(found[0])
    assert project.exists()
    with pytest.raises(ValueError, match="suggestions"):
        nx.discover(path, sounds=3, enable=[99])


def test_spectrogram_and_new_project_write_files(recording, tmp_path) -> None:
    path, _ = recording
    png = tmp_path / "s.png"
    nx.spectrogram(path, png, width=200, height=100, log=True)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    nx.new_project(tmp_path / "new.nxp")
    assert b"Background noise" in (tmp_path / "new.nxp").read_bytes()


def test_arrays_without_numpy_keep_shape(recording) -> None:
    _, samples = recording
    mono = nx.denoise_array(samples, RATE)
    assert isinstance(mono, array.array) and mono.typecode == "f"
    assert len(mono) == len(samples)
    assert rms(mono[3000:12000]) < 0.3 * rms(samples[3000:12000])
    stereo = nx.denoise_array([samples, samples], RATE)
    assert isinstance(stereo, list) and len(stereo) == 2
    assert list(stereo[0]) == list(stereo[1])


def test_arrays_with_numpy(recording, tmp_path) -> None:
    np = pytest.importorskip("numpy")
    path, samples = recording
    x = np.asarray(samples, dtype=np.float64)
    y = nx.denoise_array(x, RATE)
    assert isinstance(y, np.ndarray) and y.dtype == np.float32 and y.shape == x.shape
    # The same processing as for the file.
    out = tmp_path / "clean.wav"
    nx.denoise(path, out)
    (clean,), _ = read_wav(out)
    from_file = nx.denoise_array(nx.load(path)[0][0], RATE)
    assert np.allclose(from_file, np.asarray(clean, dtype=np.float32), atol=1e-6)
    both = nx.process_array(np.stack([x, x]), RATE)
    assert both.shape == (2, len(samples))
    loaded, rate = nx.load(path)
    assert rate == RATE and loaded.shape == (1, len(samples))


def test_errors() -> None:
    with pytest.raises(nx.Error, match="nonexistent"):
        nx.denoise("/nonexistent/in.wav", "out.wav")
    with pytest.raises(ValueError, match="rule"):
        nx.denoise_array([0.0] * 1000, RATE, rule="gate")
    with pytest.raises(ValueError, match="pair"):
        nx.mark("in.wav", time=1.0, freq=(1, 2))
    with pytest.raises(ValueError, match="same number"):
        nx.denoise_array([[0.0] * 10, [0.0] * 11], RATE)
    with pytest.raises(ValueError, match="no filter 5"):
        nx.process_array([0.0] * 1000, RATE, solo=[5])


def test_command_line_tool(recording, tmp_path) -> None:
    path, _ = recording
    out = tmp_path / "cli.wav"
    run = subprocess.run(
        [sys.executable, "-m", "noiseextractor", "denoise", str(path), "-o", str(out)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert "wrote" in run.stderr
    assert out.exists()
    bad = subprocess.run([sys.executable, "-m", "noiseextractor", "frobnicate"], capture_output=True, text=True)
    assert bad.returncode == 2
    assert "noiseextractor-cli" in bad.stderr
