# Detrending & standardization

Standardization removes the low-frequency biological growth trend from each raw
ring-width series, converting it to a dimensionless **ring-width index (RWI)**
centered near 1.0 so series of different sizes and ages can be averaged into a
chronology. dplPy offers the dplR curve-fitting methods plus regional curve,
signal-free, age-dependent-spline, and power-transform approaches.

## `detrend`: curve fitting

`detrend` fits a growth curve to each series and divides (or subtracts) it out:

```python
import dplpy as dpl

rwl = dpl.readers("ca533.rwl", header=True)
rwi = dpl.detrend(rwl, fit="Spline")
```

The `fit` argument selects the curve, using dplR's names:

| `fit` | Curve |
|---|---|
| `"Spline"` (default) | Cubic smoothing spline |
| `"ModNegExp"` | Modified negative exponential, with a negexp → linear → mean fallback chain |
| `"ModHugershoff"` | Modified Hugershoff |
| `"Mean"` | Horizontal (series mean) |
| `"AgeDepSpline"` | Age-dependent spline |

Other important arguments:

- `method` — `"ratio"` (default; divide by the curve) or `"difference"`
  (subtract). Ratios are the traditional RWI; differences suit
  variance-stabilized workflows.
- `period` / `f` — spline stiffness (wavelength) and frequency response. `period`
  defaults to a 67 %-of-length spline; a negative value is read as a *percent
  spline* (ARSTAN convention), and a value in `(0, 1]` as a fraction of length.
  `f` is the amplitude cutoff at that wavelength (default 0.5 = 50 %).
- `pos_slope` — whether to allow a positively-sloped fitted curve.
- `return_info` / `verbose` — return or print the per-series curve choices.

You can also pass a *list* of curves to try in order (as in dplR), and detrend a
single series (`pandas.Series`) as well as a whole collection.

## `rcs`: regional curve standardization

Regional curve standardization aligns series by cambial age and divides by a
single regional growth curve, preserving low-frequency (e.g. multi-centennial)
signal that curve-by-curve detrending removes:

```python
rwi = dpl.rcs(rwl, po=pith_offsets, biweight=True)
```

`po` is a pith-offset table (see [`po_to_wc`/`wc_to_po`](../reference/utilities.md)).
`preset="crust"` builds the regional curve the CRUST way (Melvin & Briffa 2014).

## `sfrcs`: signal-free RCS

Ordinary RCS builds one regional curve and detrends every series by it, but when
the sample's age structure varies over calendar time that curve absorbs part of
the common signal and biases the low frequencies ("trend distortion").
`sfrcs` breaks the feedback by iterating: each pass rebuilds the regional curve
from measurements that have had the *current chronology* divided out, while still
forming the tree indices from the original measurements.

```python
rwi = dpl.sfrcs(rwl, po=pith_offsets)
info = dpl.sfrcs(rwl, po=pith_offsets, return_info=True)  # + chronology, curve, convergence
```

It is a port of CRUST's single-curve signal-free RCS (Melvin & Briffa 2014) — a
method with **no dplR counterpart**, so it is faithful to CRUST rather than
validated against dplR. On CRUST's own sample data it reproduces a headless build
of CRUST to about 1×10⁻³.

- `biweight_curve` (default `False`) / `biweight_crn` (default `False`) — form the
  regional curve and the chronology with the arithmetic mean, matching CRUST's
  defaults; set either to `True` for a Tukey biweight robust mean.
- `ss` (default `10`) — the age-dependent spline stiffness offset for the curve
  (an 11-year minimum), as in CRUST.
- `max_iterations` (default `40`) / `tol` (default `1e-3`) — the signal-free
  iteration budget and convergence threshold (`max |Δchronology| < tol`), as in
  CRUST. Running at `max_iterations=1` performs no signal removal and reduces to
  one-pass CRUST RCS.

Signal-free RCS needs a wide spread of tree **start dates** to work well: Melvin &
Briffa (2014) show distortion of the recovered signal appears once the range of
starting years falls below about 200 years, and is considerable below about 100.
Use it on collections whose trees germinate across a broad time range, not on a
narrow single-cohort sample.

Multi-curve RCS (several regional curves with tree-to-curve allocation) is not yet
implemented.

## `ssf`: signal-free standardization

Signal-free standardization iterates detrending and chronology-building to reduce
the trend-distortion (end-effect) bias of ordinary detrending:

```python
rwi = dpl.ssf(rwl)
```

It repeats until the high-frequency chronology stops changing (median absolute
difference below `mad_threshold`). `preset="crust"` selects the CRUST variant.

## `ads` and `powt`

- **`ads`** fits an age-dependent spline to a single series — a spline whose
  stiffness increases with cambial age. It is also available inside `detrend`
  via `fit="AgeDepSpline"`.
- **`powt`** applies a power transformation (Cook & Peters) as an alternative to
  ratio detrending for variance stabilization.

## Choosing a method

Spline and `ModNegExp` are the everyday choices for site chronologies. Reach for
`rcs` when you need to retain long-timescale climate signal, `sfrcs` when you want
RCS *and* the signal-free correction for trend distortion under an uneven age
structure, `ssf` when end-effect bias matters in curve-by-curve detrending, and
`powt`/`difference` detrending when heteroscedastic variance is the concern. See the
[Detrending & standardization reference](../reference/detrending.md) for every
parameter.
