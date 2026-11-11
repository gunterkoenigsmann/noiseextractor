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
