# Golden audio fixture — CC0

`golden_30s.flac`: 30.000 s, 44.1 kHz, 2 channels, PCM_16 FLAC, 1,557,646 bytes.
SHA-256 `d6208248592cbfb20cc55ca0c6d53aacd745b69edde971c59cf657556ba4523d`.

The golden gate is CPU-vs-ROCm equivalence on real audio. This is a real vocal +
instrument mix (vocals, banjo, ukulele, acoustic guitar, harmonica, stomp box) that
may be redistributed, so no commercial recording is referenced by the tests.

## Source

- Performer: Jacob Green (The OneManGreenBand)
- Item: Jacob Green Live at HopMonk Novato Beer Garden, 2014-09-13
- Track: "01 Give it all away"
- Item page: <https://archive.org/details/jacobgreenathopmonknovato>
- File: <https://archive.org/download/jacobgreenathopmonknovato/01%20Give%20it%20all%20away.flac>
- Licence: CC0 1.0 — Public Domain Mark, declared by the uploader (Growth Productions):
  <http://creativecommons.org/publicdomain/mark/1.0/>
- Source SHA-256: `de7807b279becf41132429131a10c411884902782b303fb615f14b78e988d59e`
  (12,528,369 bytes, 244.43 s, 44.1 kHz stereo)
- Retrieved: 2026-11-09.

## How the excerpt was made

Seconds 60.000–90.000 of the source, read and written with libsndfile through
`soundfile` at the source rate — no resampling, no gain change:

```python
data, sr = soundfile.read(source, dtype="float64")
soundfile.write("golden_30s.flac", data[60*sr:90*sr], sr,
                subtype="PCM_16", format="FLAC")
```

`tests/golden/runner.golden_input()` crops the first `CROP_SECONDS` (30 s) of
`uvr_paths.INPUT_FILE`; this fixture is exactly that long, so the crop is the file.
