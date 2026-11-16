# Test recordings

Short excerpts of real field recordings, so tests also see what real
microphones, rooms and lossy sources do to sounds.

| File | Content | Source | License |
|------|---------|--------|---------|
| `owl-calls.flac` | Two calls of a little owl (*Athene noctua*), stereo, 10 s, resampled to 16 kHz | Gunter Königsmann, `2020-06-06_kautz_session.mp3` from 2:13.5, <https://www.peterpall.de/OutsideSounds/?p=archive&cat=all> | GPL-3.0-or-later, given by the author for this project |

Made with:

```sh
ffmpeg -ss 133.5 -t 10 -i 2020-06-06_kautz_session.mp3 -ar 16000 \
    -sample_fmt s16 -compression_level 12 owl-calls.flac
```

Keep new excerpts this small (a few hundred kB) and add a row here.
