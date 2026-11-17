# Test recordings

Short excerpts of real field recordings, so tests also see what real
microphones, rooms and lossy sources do to sounds.

| File | Content | Source | License |
|------|---------|--------|---------|
| `owl-calls.flac` | Two calls of a little owl (*Athene noctua*), stereo, 10 s, resampled to 16 kHz | Gunter Königsmann, `2020-06-06_kautz_session.mp3` from 2:13.5, <https://www.peterpall.de/OutsideSounds/?p=archive&cat=all> | GPL-3.0-or-later, given by the author for this project |
| `cricket-bursts.ogg` | An insect (cricket) chirping in pulse-train bursts at about 6 Hz, 6–20 kHz, stereo, 185 s, 48 kHz. Loud bursts every 6–7 s, quieter ones (more distant, or from another direction) at about 45, 52, 63, 104, 109 s and 152–178 s; birds singing at 2–6 kHz; broadband rustling at 182 s | Gunter Königsmann, recorded by the author | GPL-3.0-or-later, given by the author for this project |

Made with:

```sh
ffmpeg -ss 133.5 -t 10 -i 2020-06-06_kautz_session.mp3 -ar 16000 \
    -sample_fmt s16 -compression_level 12 owl-calls.flac
```

```sh
ffmpeg -i Grille.flac -c:a libvorbis -q:a 6 cricket-bursts.ogg
```

The cricket recording is kept whole (4.4 MB): the quiet and distant bursts
that make it useful are spread over all of it, and the pulses reach 20 kHz,
so it can't be resampled. Vorbis quality 6 keeps the channel level ratio and
delay the direction checks rely on (the marked burst measures −7.2 dB here
against −7.3 dB in the lossless original); quality 4 noticeably shifts the
pulse rate and burst extents.

Keep new excerpts as small as they can be (a few hundred kB) and add a row
here.
