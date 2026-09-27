# Artemis II flyby replay

A single-page mission dashboard: NASA's Artemis II lunar flyby (April 2026) flown under several
OrbitalEngine model tiers side by side, each scored against NASA's navigation data. It shows:

- **3D replay** (CesiumJS): an Earth-centred inertial view of every model's Orion track and NASA's,
  with the Moon at its tabulated position. There are camera presets for the whole mission, following
  Orion, the Moon and Earth, plus event markers with hover labels.
- **Mission clock**: play, pause, speed and a timeline scrubber with event ticks, jump-to buttons,
  and a live table giving each model's distance from NASA, Earth and the Moon.
- **Scorecard**: each model's closest lunar approach, farthest distance, return to Earth, lunar
  blackout and final error, with the difference from NASA as a status chip.
- **Charts** (Plotly): one per metric, one line per model, a cursor tied to the clock, click to seek,
  drag to zoom (all charts zoom together), and the off-scale rule described below.
- **Contact strip**: Deep Space Network windows per station and loss of signal behind the Moon, one
  bar per model.

**The data in `data/` is synthetic** until the engine side replaces it. `meta.json` says so, and the
page shows a hazard banner, a 3D stamp, a tag on NASA's row and a watermark on every chart for as long
as `"synthetic": true`.

## Run it

```
cd demo/artemis2
python -m http.server 8000
```

Then open <http://localhost:8000/>. Opening `index.html` straight from disk does not work, because the
page `fetch`es its CSV files. It needs network access to `cdn.jsdelivr.net` (CesiumJS 1.145.0,
Plotly 2.35.2) and Google Fonts. No Cesium ion account or token is used.

To regenerate the sample data (NumPy only, about a minute):

```
python demo/artemis2/make_sample_data.py
```

## Files

| Path | What it is |
|---|---|
| `index.html` | The page: HTML, CSS and JS in one file, no build step |
| `SCHEMA.md` | **The data contract.** The engine side writes exactly this |
| `data/` | `meta.json`, `models.csv`, `trajectory.csv`, `metrics.csv`, `events.csv`, `windows.csv` |
| `make_sample_data.py` | Writes the synthetic `data/` |
| `assets/` | Earth (Natural Earth II, stitched to one 2048x1024 image), Moon and Tycho-2 star-map textures, all from the CesiumJS package |
| `cesium/Assets/` | CesiumJS's IAU 2006 Earth-orientation tables for 2026, and `approximateTerrainHeights.json` |

## How the engine side plugs in

Write the six files in `data/` to `SCHEMA.md` and set `"synthetic": false` in `meta.json`. Nothing in
`index.html` changes:

- **Model tiers** come from `models.csv`, in its row order, with their colours. Add a row and the tier
  appears in the 3D view, the table, the scorecard, every chart and the contact strip.
- **Charts** come from the distinct `metric` keys in `metrics.csv`. The five known keys get titles and
  explanations; any other key still gets a chart. Keys ending `_error_km` or `_error_s` get the
  log-scale toggle.
- **Scorecard** rows come from `events.csv` (`closest_lunar_approach`, `max_earth_distance`,
  `entry_interface` or `return_perigee`) and `windows.csv` (`lunar_blackout`). The truth model's
  events are the reported figures every model is compared against.
- **Time span, chapters and scrubber ticks** come from the data.

The page does no physics. It interpolates positions linearly for drawing and reads every number it
shows from the files, so a sweep result can be published by writing CSV.

## How it works

**Inertial camera.** Positions arrive in Earth-centred ICRF. Every frame, the page computes the
ICRF-to-Earth-fixed rotation for the clock time: `Transforms.computeIcrfToFixedMatrix` from the
bundled IAU 2006 tables, or `computeTemeToPseudoFixedMatrix` (sidereal time only) if those are
unavailable. That rotation is used twice. It is the `modelMatrix` of the track, marker and label
collections, and of the Moon, so they are drawn in ICRF. It is also the camera's reference frame, via
`camera.lookAtTransform(transform, offset)` in `scene.postUpdate` with the offset carried over from
the previous frame. This is the pattern of Cesium's ICRF Sandcastle example. The camera and the tracks
therefore share a frame and stay still, and Earth is the only thing that turns. A camera tied to the
rotating Earth would smear the translunar track into a spiral. The frame label on the view says
which rotation is in use.

**Clock sync.** There is one clock, a `Cesium.Clock` (clamped to the data span, system-clock
multiplier). The CesiumWidget ticks it inside its render loop. A separate `requestAnimationFrame` loop
reads it about 15 times a second to update the readouts, the live table, the scrubber and the chart
cursors. A cursor is an absolutely positioned line placed from the chart's own x-range and plot-area
box, so moving it costs no Plotly relayout. Clicking a chart converts the pointer position into a
time the same way and seeks the clock, whether or not a data point is under the pointer. If CesiumJS
fails to load, a stand-in clock keeps everything else working.

**Off-scale rule.** For each chart in linear scale:

1. Take the peak of every visible model's series and sort the peaks.
2. Walk up from the smallest. Cut at the first step where one peak is more than **20 times** the
   one below it.
3. Fit the axis to the models below the cut. The truth model is never cut.

Each model above the cut keeps its real values, so its line runs off the top of the axis. An arrow
marks where it leaves and gives its peak, and a note under the chart names it and its peak. The live
table still shows its current value, and error charts have a log-scale toggle that puts every model
on one axis. In the sample, Earth only, Earth + oblateness and Earth + Moon (peaks of 443,000, 443,000
and 49,000 km) go off scale on the error chart, and the axis fits Earth + Moon + Sun (2.5 km). The
rule is re-applied whenever models are shown or hidden.

## Publishing as an artifact

The page is written to run inside the claude.ai artifact sandbox. Scripts load only from jsdelivr,
and everything else is same-origin. Two workarounds make CesiumJS load under a strict
Content-Security-Policy. Both were tested against a local CSP with no `unsafe-eval`, only `blob:`
workers and `connect-src 'self'`:

- Knockout, which is bundled inside `Cesium.js`, runs `(0,eval)("this")` at load time. A shim around
  the script tag answers that single call, then restores the real `eval`.
- `CESIUM_BASE_URL` is local (`cesium/`). Cesium then uses the worker code inlined in `Cesium.js`
  instead of cross-origin module workers, and fetches the frame tables same-origin. There is no tiled
  globe, because it needs workers to mesh tiles. Earth is a textured ellipsoid drawn like the Moon,
  with sunlight shading and without the atmosphere glow.

Under that CSP the console still shows five `WebAssembly.instantiate` rejections. They come from
decoders inside Cesium (for example meshopt) that this page never uses, and they are harmless.

Publish `index.html` with `data/*`, `assets/**` and `cesium/Assets/**` as supporting files, at the
same relative paths.

## Known limitations

- The sample is a layout fixture: a Keplerian Moon, toy dynamics and invented correction burns.
  Nothing in it is a measurement except the two NASA-reported figures in the `nasa` rows of
  `events.csv`.
- Track drawing and markers interpolate linearly between samples, so samples need to be dense near
  Earth and the Moon (see `SCHEMA.md`).
- Earth has no night-side lights or atmosphere glow, and its imagery is 2048x1024.
- The IAU 2006 tables shipped cover 2026 only. Other years fall back to the sidereal-time rotation.
- The contact strip is an input: the page does not compute visibility.
