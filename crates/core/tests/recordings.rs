//! Tests on short excerpts of real recordings (see `tests/fixtures/README.md`).

use noiseextractor_core::analysis::{analyze_selection, AnalysisParams, Selection};
use noiseextractor_core::discover::{discover, DiscoverParams};
use noiseextractor_core::engine::process_all;
use noiseextractor_core::filter::{FilterAction, FilterKind, Project};
use noiseextractor_core::noise::NoiseEstimate;
use noiseextractor_core::stft::{Stft, StftConfig};
use noiseextractor_core::AudioData;
use std::path::Path;

/// Loads a fixture recording.
fn fixture(name: &str) -> AudioData {
    AudioData::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
}

/// Mean power of channel 0 of `x` in `t0 .. t1` seconds.
fn power(x: &AudioData, t0: f64, t1: f64) -> f64 {
    let sr = x.sample_rate as f64;
    let s = &x.channels[0][(t0 * sr) as usize..(t1 * sr) as usize];
    s.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / s.len() as f64
}

#[test]
fn the_owl_calls_are_found_as_one_sound_and_can_be_extracted() {
    let audio = fixture("owl-calls.flac");
    assert_eq!(audio.channels.len(), 2);
    let mut project = Project {
        filters: Vec::new(),
        ..Project::default()
    };
    let params = DiscoverParams {
        components: 4,
        ..Default::default()
    };
    let found = discover(&audio, project.stft, &NoiseEstimate::default(), &params).unwrap();
    // All four patterns describe parts of the calls (2.5-7.5 kHz) and come
    // from the same place, so they are suggested as one sound.
    assert_eq!(found.groups.len(), 1, "{:?}", found.decomposition.labels);
    assert!(found.decomposition.directions.iter().all(Option::is_some));

    let first = project.add_discovery(found);
    let f = &mut project.filters[first];
    assert!(matches!(f.kind, FilterKind::Discovered(_)));
    f.enabled = true;
    f.action = FilterAction::Extract;
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); 2];
    process_all(&audio, &project, 256, 2, |r| {
        for (o, a) in out.iter_mut().zip(&r.audio) {
            o.extend_from_slice(a);
        }
        Ok(())
    })
    .unwrap();
    let out = AudioData::new(audio.sample_rate, out);
    // The calls (0.7-1.5 s and 8.3-9.1 s) stay, the stretch between them
    // (only background) mostly goes.
    for (t0, t1) in [(0.75, 1.4), (8.35, 9.0)] {
        let kept = power(&out, t0, t1) / power(&audio, t0, t1);
        assert!(kept > 0.7, "call at {t0} s: kept {kept:.2} of its power");
    }
    let between = power(&out, 3.0, 7.0) / power(&audio, 3.0, 7.0);
    assert!(between < 0.05, "background kept: {between:.2}");
}

#[test]
fn the_cricket_recording_loads_whole() {
    let audio = fixture("cricket-bursts.ogg");
    assert_eq!(audio.sample_rate, 48000);
    assert_eq!(audio.channels.len(), 2);
    let seconds = audio.channels[0].len() as f64 / 48000.0;
    assert!((seconds - 184.9).abs() < 0.2, "{seconds} s");
    // A loud burst (3 .. 5 s) stands far above the quiet stretch before it.
    assert!(power(&audio, 3.5, 5.0) > 100.0 * power(&audio, 0.5, 2.5));
}

#[test]
fn the_cricket_pulses_are_a_broadband_sound_without_partials() {
    // Issue #24: mark one burst (3.0 .. 5.2 s, 2 .. 20 kHz). The pulses are
    // a noise band, so there is no pitch track and no partials, and the
    // band is where the pulses are (about 7 .. 19.5 kHz; birds below).
    let audio = fixture("cricket-bursts.ogg");
    let stft = Stft::new(StftConfig::default());
    let sel = Selection {
        start_s: 3.0,
        end_s: 5.2,
        low_hz: 2000.0,
        high_hz: 20000.0,
    };
    let a = analyze_selection(
        &audio.mono(),
        &audio.channels,
        audio.sample_rate,
        &stft,
        &sel,
        &NoiseEstimate::default(),
        &AnalysisParams::default(),
    );
    let (lo, hi) = a.broadband.expect("cricket taken for a tone");
    let hz = |k: usize| StftConfig::default().bin_to_hz(k as f32, audio.sample_rate);
    assert!((5000.0..8500.0).contains(&hz(lo)), "band starts at {} Hz", hz(lo));
    assert!(hz(hi) > 18000.0, "band ends at {} Hz", hz(hi));
    assert!(a.partials.is_empty(), "{:?}", a.partials);
    assert!(a.pitch_track.iter().all(Option::is_none));
    // The rhythm matcher still sees the pulse rate (~5.9 Hz).
    let r = a
        .template
        .as_ref()
        .and_then(|t| t.rhythm.as_ref())
        .expect("no pulse rate");
    let rate = audio.sample_rate as f32 / (StftConfig::default().hop as f32 * r.period);
    assert!((5.0..7.0).contains(&rate), "pulsed at {rate} Hz");
}
