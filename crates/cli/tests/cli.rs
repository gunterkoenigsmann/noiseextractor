//! End-to-end tests of the command-line tool on synthetic recordings.

use noiseextractor_core::{AudioData, Project};
use std::path::Path;
use std::process::Command;

/// Path of the compiled binary under test.
fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_noiseextractor-cli"))
}

/// A 3 s recording: noise everywhere, a 660 Hz tone with an octave
/// overtone from 1.0 to 2.0 s.
fn write_test_audio(path: &Path) -> AudioData {
    let sr = 16000;
    let mut rng = 5u64;
    let x: Vec<f32> = (0..3 * sr)
        .map(|i| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let n = ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.04;
            let t = i as f32 / sr as f32;
            let tone = if (1.0..2.0).contains(&t) {
                let w = 2.0 * std::f32::consts::PI * 660.0 * t;
                0.4 * w.sin() + 0.2 * (2.0 * w).sin()
            } else {
                0.0
            };
            tone + n
        })
        .collect();
    let audio = AudioData::new(sr as u32, vec![x]);
    audio.save_wav(path).unwrap();
    audio
}

/// RMS of a sample range.
fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

#[test]
fn denoise_reduces_noise_and_keeps_tone() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.wav");
    let output = dir.path().join("out.wav");
    let audio = write_test_audio(&input);
    let st = bin()
        .args(["denoise"])
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap();
    assert!(st.success());
    let out = AudioData::load(&output).unwrap();
    assert_eq!(out.len(), audio.len());
    let (a, b) = (&audio.channels[0], &out.channels[0]);
    assert!(rms(&b[3000..12000]) < 0.3 * rms(&a[3000..12000]), "noise not reduced");
    assert!(rms(&b[20000..28000]) > 0.9 * rms(&a[20000..28000]), "tone damaged");
}

#[test]
fn mark_finds_overtone_and_extract_keeps_only_the_tone() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.wav");
    let project = dir.path().join("p.json");
    let output = dir.path().join("out.wav");
    let audio = write_test_audio(&input);
    let out = bin()
        .args(["mark"])
        .arg(&input)
        .args([
            "--time",
            "1.1:1.9",
            "--freq",
            "550:780",
            "--name",
            "tone",
            "--extract",
            "--save-project",
        ])
        .arg(&project)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let report = String::from_utf8_lossy(&out.stdout);
    assert!(
        report.contains("2.000 x pitch"),
        "octave missing from report:\n{report}"
    );

    // Only extract the tone: switch off the default noise reduction.
    let mut p = Project::load(&project).unwrap();
    p.filters[0].enabled = false;
    p.save(&project).unwrap();
    let st = bin()
        .arg("process")
        .arg(&input)
        .arg("--project")
        .arg(&project)
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap();
    assert!(st.success());
    let b = AudioData::load(&output).unwrap().channels.remove(0);
    let a = &audio.channels[0];
    // Outside the marked time span nothing is extracted ...
    assert!(rms(&b[3000..12000]) < 0.05 * rms(&a[3000..12000]));
    // ... inside, the tone (and its overtone) survive.
    assert!(rms(&b[20000..28000]) > 0.85 * rms(&a[20000..28000]));
}

#[test]
fn spectrogram_and_new_project_write_files() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.wav");
    write_test_audio(&input);
    let png = dir.path().join("s.png");
    let st = bin()
        .arg("spectrogram")
        .arg(&input)
        .arg("-o")
        .arg(&png)
        .args(["--width", "200", "--height", "100"])
        .status()
        .unwrap();
    assert!(st.success());
    assert!(std::fs::metadata(&png).unwrap().len() > 100);
    let proj = dir.path().join("new.json");
    assert!(bin().arg("new-project").arg(&proj).status().unwrap().success());
    assert_eq!(Project::load(&proj).unwrap().filters.len(), 1);
}

#[test]
fn bad_input_gives_an_error_not_a_panic() {
    let out = bin()
        .args(["process", "/nonexistent.wav", "-o", "/tmp/x.wav"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).starts_with("error:"));
}

/// Three identical rising chirps (with an overtone) at 0.4, 1.4 and 2.4 s
/// in light noise.
fn write_repeated_calls(path: &Path) -> AudioData {
    let sr = 16000;
    let mut rng = 9u64;
    let mut x: Vec<f32> = (0..3 * sr)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.01
        })
        .collect();
    for start in [0.4f32, 1.4, 2.4] {
        let s = (start * sr as f32) as usize;
        let n = (0.35 * sr as f32) as usize;
        let mut phase = 0.0f32;
        for i in 0..n {
            let t = i as f32 / n as f32;
            phase += 2.0 * std::f32::consts::PI * (900.0 + 700.0 * t) / sr as f32;
            let env = (std::f32::consts::PI * t).sin();
            x[s + i] += env * (0.3 * phase.sin() + 0.12 * (2.0 * phase).sin());
        }
    }
    let audio = AudioData::new(sr as u32, vec![x]);
    audio.save_wav(path).unwrap();
    audio
}

#[test]
fn marking_one_call_suppresses_all_repetitions() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("calls.wav");
    let project = dir.path().join("p.json");
    let output = dir.path().join("out.wav");
    let audio = write_repeated_calls(&input);
    let out = bin()
        .arg("mark")
        .arg(&input)
        .args(["--time", "0.4:0.75", "--freq", "850:1650", "--save-project"])
        .arg(&project)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let report = String::from_utf8_lossy(&out.stdout);
    assert!(report.contains("same sound found 3 times"), "report:\n{report}");
    assert!(report.contains("the marked one"), "report:\n{report}");

    let st = bin()
        .arg("process")
        .arg(&input)
        .arg("--project")
        .arg(&project)
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap();
    assert!(st.success());
    let b = AudioData::load(&output).unwrap().channels.remove(0);
    let a = &audio.channels[0];
    for start in [0.4f32, 1.4, 2.4] {
        // middle of each call
        let r = ((start + 0.1) * 16000.0) as usize..((start + 0.25) * 16000.0) as usize;
        assert!(
            rms(&b[r.clone()]) < 0.2 * rms(&a[r.clone()]),
            "call at {start} s survived"
        );
    }
}

/// Stereo version of [`write_repeated_calls`]: the calls at 0.4 and 1.4 s
/// come from the left (right channel 8 dB quieter and 3 samples later), the
/// identical call at 2.4 s from the right (left channel 8 dB quieter).
fn write_stereo_calls(path: &Path) -> AudioData {
    let sr = 16000;
    let len = 3 * sr;
    let mut rng = 11u64;
    let mut ch: Vec<Vec<f32>> = (0..2)
        .map(|_| {
            (0..len)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.01
                })
                .collect()
        })
        .collect();
    let quiet = 10f32.powf(-8.0 / 20.0);
    for (start, gains, delays) in [
        (0.4f32, [1.0, quiet], [0, 3]),
        (1.4, [1.0, quiet], [0, 3]),
        (2.4, [quiet, 1.0], [0, 0]),
    ] {
        let s = (start * sr as f32) as usize;
        let n = (0.35 * sr as f32) as usize;
        let mut phase = 0.0f32;
        for i in 0..n {
            let t = i as f32 / n as f32;
            phase += 2.0 * std::f32::consts::PI * (900.0 + 700.0 * t) / sr as f32;
            let env = (std::f32::consts::PI * t).sin();
            let v = env * (0.3 * phase.sin() + 0.12 * (2.0 * phase).sin());
            for c in 0..2 {
                ch[c][s + i + delays[c]] += gains[c] * v;
            }
        }
    }
    let audio = AudioData::new(sr as u32, ch);
    audio.save_wav(path).unwrap();
    audio
}

#[test]
fn stereo_repetition_from_another_place_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("stereo.wav");
    let project = dir.path().join("p.json");
    let output = dir.path().join("out.wav");
    let audio = write_stereo_calls(&input);
    let out = bin()
        .arg("mark")
        .arg(&input)
        .args(["--time", "0.4:0.75", "--freq", "850:1650", "--save-project"])
        .arg(&project)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let report = String::from_utf8_lossy(&out.stdout);
    assert!(
        report.contains("direction: channel 1 vs 0: level -8."),
        "report:\n{report}"
    );
    assert!(report.contains("same sound found 2 times"), "report:\n{report}");

    let st = bin()
        .arg("process")
        .arg(&input)
        .arg("--project")
        .arg(&project)
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap();
    assert!(st.success());
    let b = AudioData::load(&output).unwrap();
    // Louder channel of each call: left for the first two, right for the third.
    for (start, c, removed) in [(0.4f32, 0, true), (1.4, 0, true), (2.4, 1, false)] {
        let r = ((start + 0.1) * 16000.0) as usize..((start + 0.25) * 16000.0) as usize;
        let ratio = rms(&b.channels[c][r.clone()]) / rms(&audio.channels[c][r]);
        if removed {
            assert!(ratio < 0.2, "call at {start} s survived ({ratio})");
        } else {
            assert!(
                ratio > 0.8,
                "call from the other place at {start} s was removed ({ratio})"
            );
        }
    }
}

/// A 3 s mono recording of two sounds taking turns over faint noise: a
/// 660 Hz tone with an octave overtone at 0.3-1.0 s and 1.8-2.4 s, and a
/// warbling 3 kHz whistle at 1.1-1.7 s and 2.5-2.9 s.
fn write_two_sounds(path: &Path) -> AudioData {
    let sr = 16000;
    let mut rng = 9u64;
    let x: Vec<f32> = (0..3 * sr)
        .map(|i| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let n = ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.01;
            let t = i as f32 / sr as f32;
            let w = 2.0 * std::f32::consts::PI * t;
            let tone = if (0.3..1.0).contains(&t) || (1.8..2.4).contains(&t) {
                0.3 * (660.0 * w).sin() + 0.15 * (1320.0 * w).sin()
            } else {
                0.0
            };
            let whistle = if (1.1..1.7).contains(&t) || (2.5..2.9).contains(&t) {
                0.3 * (3000.0 * w + 3.0 * (6.0 * w).sin()).sin()
            } else {
                0.0
            };
            tone + whistle + n
        })
        .collect();
    let audio = AudioData::new(sr as u32, vec![x]);
    audio.save_wav(path).unwrap();
    audio
}

#[test]
fn discovered_sound_can_be_extracted() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.wav");
    let project = dir.path().join("p.json");
    let output = dir.path().join("out.wav");
    let audio = write_two_sounds(&input);
    let out = bin()
        .args(["discover", "--sounds", "3"])
        .arg(&input)
        .arg("--save-project")
        .arg(&project)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let report = String::from_utf8_lossy(&out.stdout);
    assert!(report.contains("suggestion 1"), "{report}");

    // Switch on, as an extract filter, the suggestion with the tone.
    let mut p = Project::load(&project).unwrap();
    let tone = p
        .filters
        .iter()
        .position(|f| {
            f.suggested
                && f.parts()
                    .iter()
                    .any(|(l, _)| l.contains("peak 656 Hz") || l.contains("peak 664 Hz"))
        })
        .unwrap_or_else(|| panic!("no suggestion with the tone:\n{report}"));
    p.filters[tone].enabled = true;
    p.filters[tone].action = noiseextractor_core::FilterAction::Extract;
    // Keep the background noise reduction out of the way.
    p.filters
        .retain(|f| !matches!(f.kind, noiseextractor_core::FilterKind::NoiseReduction(_)));
    p.save(&project).unwrap();
    let st = bin()
        .arg("process")
        .arg(&input)
        .arg("--project")
        .arg(&project)
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap();
    assert!(st.success());
    let result = AudioData::load(&output).unwrap();
    let (a, b) = (&audio.channels[0], &result.channels[0]);
    let range = |t0: f32, t1: f32| ((t0 * 16000.0) as usize, (t1 * 16000.0) as usize);
    let (t0, t1) = range(0.4, 0.9);
    assert!(rms(&b[t0..t1]) > 0.8 * rms(&a[t0..t1]), "tone damaged");
    let (w0, w1) = range(1.2, 1.6);
    assert!(rms(&b[w0..w1]) < 0.2 * rms(&a[w0..w1]), "whistle kept");
}

/// Runs `process` with extra arguments and returns the first channel.
fn process_with(input: &Path, project: &Path, output: &Path, extra: &[&str]) -> Vec<f32> {
    let out = bin()
        .arg("process")
        .arg(input)
        .arg("--project")
        .arg(project)
        .arg("-o")
        .arg(output)
        .args(extra)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    AudioData::load(output).unwrap().channels.remove(0)
}

#[test]
fn solo_and_without_split_the_recording_at_the_marked_tone() {
    // The tone is marked as a sound to *suppress*; solo plays it anyway,
    // "without" plays the rest. Noise reduction is off so that the two
    // add up (in power) to the recording.
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.wav");
    let project = dir.path().join("p.json");
    let audio = write_test_audio(&input);
    let st = bin()
        .arg("mark")
        .arg(&input)
        .args(["--time", "1.1:1.9", "--freq", "550:780", "--save-project"])
        .arg(&project)
        .output()
        .unwrap();
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let mut p = Project::load(&project).unwrap();
    p.filters[0].enabled = false;
    p.save(&project).unwrap();

    let solo = process_with(&input, &project, &dir.path().join("solo.wav"), &["--solo", "2"]);
    let without = process_with(&input, &project, &dir.path().join("without.wav"), &["--without", "2"]);
    let a = &audio.channels[0];
    let (tone, noise) = (20000..28000, 3000..12000);
    // The filter suppresses already, so "without" is the plain output.
    let plain = process_with(&input, &project, &dir.path().join("plain.wav"), &[]);
    assert!(without.iter().zip(&plain).all(|(x, y)| (x - y).abs() < 1e-6));
    // Solo: the tone, and nothing before it.
    assert!(rms(&solo[tone.clone()]) > 0.85 * rms(&a[tone.clone()]));
    assert!(rms(&solo[noise.clone()]) < 0.05 * rms(&a[noise.clone()]));
    // Without: the noise before the tone untouched, the fundamental and
    // its octave gone. Leaving the octave alone would keep 0.45 of the
    // tone's RMS (issue #16), so 0.15 checks that the octave is removed too.
    assert!(rms(&without[noise.clone()]) > 0.95 * rms(&a[noise]));
    let left = rms(&without[tone.clone()]) / rms(&a[tone.clone()]);
    assert!(left < 0.15, "without keeps {left} of the tone");
    // Together (in power) they are the recording.
    let sum = (rms(&solo[tone.clone()]).powi(2) + rms(&without[tone.clone()]).powi(2)).sqrt();
    let ratio = sum / rms(&a[tone]);
    assert!((0.9..1.1).contains(&ratio), "solo + without = {ratio} x the recording");
}

#[test]
fn solo_of_two_filters_plays_both_sounds() {
    // The tone's fundamental and its octave marked as two filters: each
    // solo keeps one of them, "--solo 2,3" keeps both.
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.wav");
    let project = dir.path().join("p.json");
    let audio = write_test_audio(&input);
    for (i, freq) in ["550:780", "1200:1450"].into_iter().enumerate() {
        let mut cmd = bin();
        cmd.arg("mark").arg(&input).args(["--time", "1.1:1.9", "--freq", freq]);
        if i > 0 {
            cmd.arg("--project").arg(&project);
        }
        let out = cmd.arg("--save-project").arg(&project).output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }
    let p = Project::load(&project).unwrap();
    assert_eq!(p.filters.len(), 3, "noise reduction + two marks");
    let tone = 20000..28000;
    let energy = |x: &[f32]| rms(&x[tone.clone()]).powi(2);
    let low = process_with(&input, &project, &dir.path().join("low.wav"), &["--solo", "2"]);
    let high = process_with(&input, &project, &dir.path().join("high.wav"), &["--solo", "3"]);
    let both = process_with(&input, &project, &dir.path().join("both.wav"), &["--solo", "2,3"]);
    // The two picks are heard together: their energies add up (they
    // hardly overlap), and together they are the whole tone.
    let input_energy = rms(&audio.channels[0][tone.clone()]).powi(2);
    assert!(energy(&low) > 0.3 * input_energy && energy(&high) > 0.2 * input_energy);
    let sum = energy(&low) + energy(&high);
    assert!((energy(&both) - sum).abs() < 0.1 * sum, "{} vs {sum}", energy(&both));
    assert!(energy(&both) > 0.9 * input_energy);
    // Removing both leaves next to nothing of the tone.
    let without = process_with(&input, &project, &dir.path().join("w.wav"), &["--without", "2,3"]);
    assert!(energy(&without) < 0.01 * input_energy);
}

#[test]
fn solo_of_a_missing_filter_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.wav");
    write_test_audio(&input);
    let proj = dir.path().join("p.json");
    assert!(bin().arg("new-project").arg(&proj).status().unwrap().success());
    let out = bin()
        .arg("process")
        .arg(&input)
        .arg("--project")
        .arg(&proj)
        .arg("-o")
        .arg(dir.path().join("out.wav"))
        .args(["--solo", "2"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("no filter 2"));
}

#[test]
fn mark_reports_insect_pulses_as_broadband_without_partials() {
    // Issue #24: the cricket's noise pulses used to get a "pitch" jumping
    // between 8 and 19 kHz and eight made-up partials.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/tests/fixtures/cricket-bursts.ogg");
    let out = bin()
        .arg("mark")
        .arg(&fixture)
        .args([
            "--time",
            "3.0:5.2",
            "--freq",
            "2000:20000",
            "--name",
            "cricket",
            "--extract",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let report = String::from_utf8_lossy(&out.stdout);
    assert!(report.contains("broadband sound"), "{report}");
    assert!(report.contains("pulsed at"), "{report}");
    assert!(!report.contains("dominant frequency"), "{report}");
    assert!(!report.contains("related partial"), "{report}");
}

#[test]
fn process_writes_every_export_format() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.wav");
    let audio = write_test_audio(&input);
    // Opus is checked in the core crate's tests: Symphonia cannot decode it.
    for ext in ["flac", "ogg", "mp3", "m4a"] {
        let output = dir.path().join(format!("out.{ext}"));
        let st = bin()
            .arg("process")
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .status()
            .unwrap();
        assert!(st.success(), "{ext}");
        let out = AudioData::load(&output).unwrap();
        assert_eq!(out.sample_rate, 16000, "{ext}");
        // Lossy codecs may add encoder delay or padding at the end.
        assert!(
            out.len() >= audio.len() && out.len() < audio.len() + 4000,
            "{ext}: {}",
            out.len()
        );
        // The tone survives the denoising and the encoding.
        let tone = rms(&out.channels[0][20000..28000]);
        assert!(tone > 0.25, "{ext}: tone rms {tone}");
    }
    let opus = dir.path().join("out.opus");
    let st = bin()
        .arg("process")
        .arg(&input)
        .args(["--bitrate", "48", "-o"])
        .arg(&opus)
        .status()
        .unwrap();
    assert!(st.success());
    assert!(std::fs::read(&opus).unwrap().starts_with(b"OggS"));
}

#[test]
fn explicit_format_and_unknown_extension() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.wav");
    write_test_audio(&input);
    let output = dir.path().join("out.bin");
    let out = bin()
        .arg("process")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--format"));
    assert!(!output.exists(), "nothing is written before the format is known");
    let st = bin()
        .arg("process")
        .arg(&input)
        .args(["--format", "flac", "-o"])
        .arg(&output)
        .status()
        .unwrap();
    assert!(st.success());
    assert!(std::fs::read(&output).unwrap().starts_with(b"fLaC"));
}
