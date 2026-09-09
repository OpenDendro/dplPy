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
`rcs` when you need to retain long-timescale climate signal, `ssf` when
end-effect bias matters, and `powt`/`difference` detrending when heteroscedastic
variance is the concern. See the
[Detrending & standardization reference](../reference/detrending.md) for every
parameter.
