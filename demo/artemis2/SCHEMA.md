# Artemis II dashboard: data contract

The dashboard (`index.html`) reads six files from `demo/artemis2/data/`. The engine side writes them;
the page does no physics of its own. It only interpolates, plots, and compares values against the
truth model. Anything that follows this contract renders without code changes, including new model
tiers, new metrics and new events.

`make_sample_data.py` writes a **synthetic** instance of every file. `meta.json` says so, and the page
shows a banner while it does.

## Conventions

- **Time.** Every time is `t_s`: seconds (float) from the epoch in `meta.json`. It is the TDB-free
  offset: `epoch_utc + t_s` is the UTC instant the page displays. Negative values are allowed.
  Sample times need not be uniform or shared between models.
- **Frame.** Positions are Earth-centred ICRF (J2000 axes), in km. The page rotates them into
  Earth-fixed coordinates for drawing. Do not send Earth-fixed or TEME positions.
- **Units.** km, km/s, m/s and seconds. The `unit` column is displayed as given.
- **CSV.** UTF-8, one header row, comma separated, RFC 4180 quoting (a field containing a comma or
  quote is wrapped in `"` with inner quotes doubled). `\n` or `\r\n` line endings. Column order is
  free; the page reads columns by header name. Unknown extra columns are ignored.
- **Empty values.** An empty `value` means "not applicable" (a milestone event has no value).
- **Missing data.** Do not write `nan` or `inf`. Leave the row out.

## `meta.json`

```json
{
  "synthetic": false,
  "title": "Artemis II lunar flyby",
  "epoch_utc": "2026-04-01T22:35:00.000Z",
  "epoch_tdb": "2026-04-01T22:36:09.184 TDB",
  "epoch_label": "Launch (T+0)",
  "frame": "Earth-centred ICRF (J2000 axes), km",
  "source": "JPL Horizons Orion (-1024) ephemeris, retrieved 2026-04-12; OrbitalEngine sweep",
  "truth_source": "NASA/JPL reconstructed Orion trajectory",
  "generated_utc": "2026-05-01T12:00:00Z",
  "engine_commit": "764acb8",
  "engine_version": "0.9.0",
  "notes": ["Free-text lines shown in the provenance footer."]
}
```

| Key | Required | Meaning |
|---|---|---|
| `synthetic` | yes | `true` shows the SYNTHETIC SAMPLE DATA banner and watermarks. The engine writes `false`. |
| `epoch_utc` | yes | ISO 8601 UTC with `Z`. `t_s = 0` is this instant. |
| `epoch_tdb` | yes | The same instant in TDB, for the record. The page displays UTC. |
| `epoch_label` | no | What `t_s = 0` is. Mission elapsed time (`T+`) is measured from it. |
| `title` | no | Heading text. Defaults to "Artemis II lunar flyby". |
| `frame` | no | Shown in the footer. |
| `source`, `truth_source` | yes | Provenance, shown in the footer. |
| `generated_utc`, `engine_commit`, `engine_version` | yes (`engine_version` may be `null`) | Provenance. |
| `notes` | no | List of strings, shown in the footer. |

## `models.csv`

One row per model tier, in display order (legend, cards and chart traces follow it).

| Column | Type | Meaning |
|---|---|---|
| `model_id` | string, `[a-z0-9_]+` | Key used by every other file. |
| `label` | string | Short display name ("Earth + Moon"). |
| `description` | string | One plain-English sentence on what the tier includes. Shown on its card. |
| `colour` | `#rrggbb` | Line colour on light backgrounds. |
| `colour_dark` | `#rrggbb`, optional | Line colour on dark backgrounds and in the 3D view. Defaults to `colour`. |
| `is_truth` | `0` or `1` | Exactly one row is `1`. It is the reference every delta is taken against, and is drawn emphasised. |

Sample: `nasa` (truth), `kepler`, `j2`, `moon`, `moon_sun`. Pick colours that clear 3:1 against both
page surfaces (`#ffffff` light, `#0e1520` dark) and against black space.

## `trajectory.csv`

| Column | Type | Meaning |
|---|---|---|
| `t_s` | float | Seconds from epoch. Rows for one `(model_id, body)` must be in increasing `t_s`. |
| `model_id` | string | From `models.csv`. |
| `body` | `orion` or `moon` | Which body the row places. |
| `x_km`, `y_km`, `z_km` | float | Earth-centred ICRF position, km. Three decimals are plenty. |

- `orion` rows are required for every model over the span the model was flown. A model may start
  later than the truth (it is seeded from it) and may end earlier or later.
- `moon` rows are required for the **truth model only**. They are the Moon ephemeris the 3D view uses
  for every model. A model may add its own `moon` rows; the page ignores them today.
- The page interpolates linearly between samples, both for the moving markers and for the drawn
  track. Sample densely where the path curves: **60 s within 60,000 km of Earth or 40,000 km of the
  Moon, 300 s elsewhere** is what the sample uses. The Moon at 600 s is within 1 km of its arc.
- Expected size for a 9-day mission with five models: about **20,000 rows, 1.1 MB**
  (sample: 20,094 rows). Keep the whole data set under 8 MB.

## `metrics.csv`

Tidy long format: one row per `(t_s, model_id, metric)`.

| Column | Type | Meaning |
|---|---|---|
| `t_s` | float | Seconds from epoch. |
| `model_id` | string | From `models.csv`. |
| `metric` | string | Metric key, see below. Each distinct key becomes one chart. |
| `value` | float | Value in `unit`. |
| `unit` | string | Display unit. Must be constant per metric. |

Known keys get a title, an explanation and a sensible default axis. An unknown key still gets a
chart, titled with the key itself.

| Key | Unit | Who writes it | Meaning |
|---|---|---|---|
| `position_error_km` | km | every non-truth model | Distance between the model's Orion and the truth's Orion at the same instant. Only where both exist. Gets the log-scale toggle. |
| `earth_range_km` | km | every model | Distance from Earth's centre. |
| `moon_range_km` | km | every model | Distance from the Moon's centre (the model's own Moon if it has one, else the truth ephemeris). |
| `speed_km_s` | km/s | every model | Inertial speed relative to Earth. |
| `delta_v_m_s` | m/s | truth, and any model that plans burns | Cumulative Δv spent, a step function. |

Any key ending in `_error_km` or `_error_s` also gets the log-scale toggle. Sample at 120 s near a
body and 600 s elsewhere (sample: 37,423 rows, 1.6 MB).

## `events.csv`

| Column | Type | Meaning |
|---|---|---|
| `t_s` | float | When it happens (or is predicted to) under that model. |
| `model_id` | string | Whose event. Reported events belong to the truth model (`nasa`). |
| `event` | string | Event key. The same key across models makes them comparable. |
| `kind` | `burn`, `apsis`, `milestone` | `burn`: `value` is the burn Δv. `apsis`: `value` is the distance quantity named by the event. `milestone`: `value` optional. |
| `value` | float or empty | In `unit`. |
| `unit` | string or empty | |
| `note` | string | One plain-English line, shown on hover. |

Keys the page treats specially (all others are listed and marked on the track generically):

| Key | Kind | `value` | Used for |
|---|---|---|---|
| `closest_lunar_approach` | apsis | altitude above the mean lunar radius (1,737.4 km), km | Metric cards: "Closest to the Moon", delta vs truth in km and in time. |
| `max_earth_distance` | apsis | distance from Earth's centre, km | Metric cards: "Farthest from Earth". |
| `entry_interface` | milestone | altitude, km (121.92) | Metric cards: "Entry interface" time vs truth. A model without it did not reach the atmosphere. |
| `return_perigee` | apsis | altitude, km | Metric cards: shown instead of entry when a model misses Earth. |
| `model_seed` | milestone | none | Marks where the models start from truth. Drawn on every chart. |

Events outside a model's trajectory span are listed but not drawn on the track.

## `windows.csv`

| Column | Type | Meaning |
|---|---|---|
| `model_id` | string | Whose prediction. |
| `kind` | `dsn_contact` or `lunar_blackout` | Contact window with a ground station, or loss of signal behind the Moon. |
| `station` | string | Station name for `dsn_contact` (`Goldstone`, `Madrid`, `Canberra`, any others become new rows). Empty for `lunar_blackout`. |
| `start_s`, `end_s` | float | Window edges, seconds from epoch. |

The sample uses a 10° elevation mask and a spherical Earth, with lunar blackout taken as the Moon's
disc covering Orion as seen from Earth's centre. The engine side should state its own definitions in
`meta.json` `notes`.

## Size budget

The page itself is about 90 KB plus 1.4 MB of bundled textures. CesiumJS and Plotly load from
`cdn.jsdelivr.net`. Data should stay under 8 MB so the whole artifact stays well under its 16 MB
limit; the sample is 2.7 MB.
