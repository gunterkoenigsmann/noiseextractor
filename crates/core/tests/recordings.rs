//! Tests on short excerpts of real recordings (see `tests/fixtures/README.md`).

use noiseextractor_core::analysis::{analyze_selection, AnalysisParams, Selection};
use noiseextractor_core::discover::{discover, DiscoverParams};
use noiseextractor_core::engine::process_all;
use noiseextractor_core::filter::{FilterAction, FilterKind, Project};
use noiseextractor_core::noise::NoiseEstimate;
use noiseextractor_core::similar::{find_in_range, SimilarParams};
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
        outline: Vec::new(),
    };
    let a = analyze_selection(
        audio.mono(),
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

#[test]
fn the_loud_cricket_bursts_come_from_the_marked_place() {
    // Issue #22: the loud bursts are the marked insect and must agree on
    // the direction. The quiet ones at 45, 52 and 63 s measure about
    // -5 dB against -7.2 dB, consistently in every sub-band and at 20 dB
    // above the noise, so they are taken to come from another place
    // (decided by the maintainer) and are only required to be found.
    let audio = fixture("cricket-bursts.ogg");
    let cfg = StftConfig::default();
    let stft = Stft::new(cfg);
    let mono = audio.mono();
    let sel = Selection {
        start_s: 3.0,
        end_s: 5.2,
        low_hz: 2000.0,
        high_hz: 20000.0,
        outline: Vec::new(),
    };
    let a = analyze_selection(
        mono,
        &audio.channels,
        audio.sample_rate,
        &stft,
        &sel,
        &NoiseEstimate::default(),
        &AnalysisParams::default(),
    );
    let tpl = a.template.expect("template");
    let total = cfg.frame_count(mono.len()) as i64;
    let found = find_in_range(&tpl, mono, &audio.channels, &stft, 0..total, &SimilarParams::default());
    let at = |t: f64| {
        found.iter().find(|o| {
            let start = o.frame as f64 * cfg.hop as f64 / audio.sample_rate as f64;
            let end = o.end() as f64 * cfg.hop as f64 / audio.sample_rate as f64;
            start - 0.5 <= t && t <= end
        })
    };
    for t in [9.5, 18.5, 31.0, 57.0, 70.5, 76.5, 83.5, 90.0, 96.5] {
        let o = at(t).unwrap_or_else(|| panic!("loud burst at {t} s not found"));
        let place = o.spatial_agreement.expect("stereo");
        assert!(place > 0.95, "burst at {t} s: same place {place:.2}");
    }
    for t in [45.0, 52.0, 63.0] {
        assert!(at(t).is_some(), "quiet burst at {t} s not found");
    }
}

#[test]
fn cricket_bursts_are_found_whole_and_without_blips() {
    // Issue #23, direction ignored: each real burst is found once and
    // about as long as it is, and nothing lasting only one matching window
    // (faint pulsing before 30 s, rustling at 182 s) is reported. The
    // bursts at 173 and 177 s pulse at 1-2 Hz and come from almost straight
    // ahead, so they are not taken for this cricket.
    let audio = fixture("cricket-bursts.ogg");
    let cfg = StftConfig::default();
    let stft = Stft::new(cfg);
    let mono = audio.mono();
    let sel = Selection {
        start_s: 3.0,
        end_s: 5.2,
        low_hz: 2000.0,
        high_hz: 20000.0,
        outline: Vec::new(),
    };
    let a = analyze_selection(
        mono,
        &audio.channels,
        audio.sample_rate,
        &stft,
        &sel,
        &NoiseEstimate::default(),
        &AnalysisParams::default(),
    );
    let tpl = a.template.expect("template");
    let params = SimilarParams {
        spatial_weight: 0.0,
        ..Default::default()
    };
    let total = cfg.frame_count(mono.len()) as i64;
    let seconds = |f: i64| f as f64 * cfg.hop as f64 / audio.sample_rate as f64;
    let spans: Vec<(f64, f64)> = find_in_range(&tpl, mono, &audio.channels, &stft, 0..total, &params)
        .iter()
        .map(|o| (seconds(o.frame), seconds(o.end())))
        .collect();
    for (s, e) in &spans {
        assert!(e - s > 1.3, "blip {s:.1}..{e:.1} s in {spans:?}");
        assert!(
            !(26.0..29.5).contains(s) && *s < 180.0,
            "spurious hit at {s:.1} s in {spans:?}"
        );
    }
    for t in [
        4.0, 10.0, 19.0, 31.5, 45.5, 52.0, 63.5, 104.0, 109.5, 153.5, 160.5, 167.5,
    ] {
        let n = spans.iter().filter(|(s, e)| *s <= t && t <= *e).count();
        assert_eq!(n, 1, "burst at {t} s found {n} times in {spans:?}");
    }
}
