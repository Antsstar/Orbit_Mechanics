# Artemis III rendezvous planners (prospective demo)

A single-page, no-build dashboard for the prospective Artemis III scenario in
`src/orbital_engine/artemis3.py`: Orion's final rendezvous transfer to the lander test vehicle,
planned by four models of rising fidelity (two-body/Lambert, + J2, + J3-J6 and 4x4 tesserals,
+ drag) and each plan flown in the most complete model. Only the orbit and mission sequence are
public; everything else is an assumption listed on the page.

## Run

```
cd demo/artemis3
python -m http.server 8000      # then open http://localhost:8000/
```

The page needs a server (it `fetch`es its data relatively) and internet for Plotly (jsDelivr) and
Google Fonts. Append `?theme=dark` or `?theme=light` to force a theme.

## Data

- `data/summary.json`: title, public facts, assumptions, transfer time, per-planner burns, miss, half-way fix.
- `data/approach.csv`: `t_s, planner, case, r_m, s_m, w_m`; Orion relative to the lander in the lander's RSW frame.

Regenerate from the repo root with `src` on the path:

```
python -c "from orbital_engine import artemis3 as A; A.export_demo(A.run(), 'demo/artemis3/data')"
```
