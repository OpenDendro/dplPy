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

The `fit` argument selects the curve (dplR's names are the canonical spelling,
and are case-insensitive):

| `fit` | Curve |
|---|---|
| `"Spline"` (default) | Cubic smoothing spline |
| `"AgeDepSpline"` | Age-dependent (increasing-stiffness) spline |
| `"ModNegExp"` | Negative exponential `a·exp(b·t)+k`, fit by **nonlinear least squares** (dplR), with a negexp → linear → mean fallback chain |
| `"NegExp"` | The **same** negative-exponential curve, fit by ARSTAN's **deterministic** method (Ed Cook's `curve`) |
| `"ModHugershoff"` | Hugershoff `a·t^b·exp(c·t)+d`, fit by **nonlinear least squares** (dplR) |
| `"Hugershoff"` | The Hugershoff curve, fit by ARSTAN's **log-linearised closed form** (Ed Cook's `hughdi`) |
| `"GeneralExp"` | General exponential `a·t·exp(b·t)` — Hugershoff with the power fixed at 1 (ARSTAN option 8) |
| `"LinearAny"` (alias `"Linear"`) | Best-fit straight line, any slope (ARSTAN option 4) |
| `"LinearNegative"` | Best-fit line constrained to a non-positive slope, falling back to the mean if the slope is positive (ARSTAN option 5) |
| `"Mean"` | Horizontal line at the series mean |

Two of these come as **deterministic/nonlinear pairs** — `NegExp`/`ModNegExp`
and `Hugershoff`/`ModHugershoff` — pairing ARSTAN's deterministic fit with dplR's
nonlinear-least-squares fit of the same curve. The two behave differently in
each case:

- **Negative exponential.** Both members minimise the *same* data-space error for
  the *same* model, so when the nonlinear fit converges `NegExp` and `ModNegExp`
  give the same curve. `NegExp`'s advantage is robustness: being deterministic it
  never fails to converge, so it still returns a curve on series where the nls
  diverges and `ModNegExp` falls back to a line or the mean — handy for batch
  processing.
- **Hugershoff.** Here the two *genuinely differ*: dplR's `ModHugershoff`
  minimises data-space error, while ARSTAN's `Hugershoff` linearises the model and
  minimises log-space error, so the fitted curves are not the same even when both
  succeed.

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

!!! warning "Internal gaps and position-based fits"

    dplPy fits every curve to the series with its missing years (`NaN`) dropped,
    so a `-9.99`-style gap value can never corrupt a fit. **`Spline`** fits a
    smoothing spline over the real calendar years, so it spans an interior gap
    correctly in time, and **`Mean`** is order-invariant — neither is affected by
    a gap. Every *other* fit is fitted against cambial age reconstructed as
    `1, 2, …, n` over the surviving rings: the parametric growth curves
    (`ModNegExp`, `NegExp`, `ModHugershoff`, `Hugershoff`, `GeneralExp`), the
    `LinearAny` / `LinearNegative` fits, and **`AgeDepSpline`** (whose stiffness
    tracks ring position). For these, an interior gap compresses the age axis
    after the gap — a ring at true cambial age 50 is fit as age 45 if five earlier
    rings are missing. For a smooth growth curve this is a minor, benign
    distortion (not the catastrophe a sentinel value would cause), but it is not
    exact. `detrend` warns when a position-based fit meets a series with interior
    gaps; you can then switch to `fit="Spline"` or infill first with
    [`fill_internal`](../reference/utilities.md) (e.g. `fill="ARSTAN"`, Ed Cook's
    opt-in regional-signal imputation) — dplPy never fills silently. See
    [Handling internal gaps](internal-gaps.md).


## `rcs`: regional curve standardization

Regional curve standardization aligns series by cambial age and divides by a
single regional growth curve, preserving low-frequency (e.g. multi-centennial)
signal that curve-by-curve detrending removes:

```python
rwi = dpl.rcs(rwl, po=pith_offsets, biweight=True)             # default AgeDepSpline, ratios
rwi = dpl.rcs(rwl, po=pith_offsets, fit="Spline", period=0.1)  # fixed 10%-of-length spline
```

The regional curve is chosen with `fit=` — the same curve vocabulary as
`detrend()`: `"AgeDepSpline"` (the default, and ARSTAN's recommended RCS curve),
`"Spline"`, the parametric growth curves (`"ModNegExp"`, `"Hugershoff"`, …), or
`None` for the unsmoothed age-aligned mean. `method=` selects the arithmetic
(`"ratio"`, the default, or `"difference"`), `period=` sets the `"Spline"`
stiffness (a fraction of the curve length, default `0.1`; `-10` is the ARSTAN
percent-of-length spline), and `pos_slope=False` (the default) keeps the old-age
end of the curve from rising. `po` is a pith-offset table (see
[`po_to_wc`/`wc_to_po`](../reference/utilities.md)); `preset="crust"` builds the
regional curve the CRUST way (Melvin & Briffa 2014).

!!! warning "Changed in 0.8.0"

    `rcs()` and `sfrcs()` inputs were harmonized with `detrend()`: the curve is
    now `fit=` (was `method="caps"`/`"ads"`), the arithmetic is `method=`
    (`"ratio"`/`"difference"`, was the boolean `ratios=`), stiffness is `period=`
    (was `nyrs=`), the default curve is `fit="AgeDepSpline"` (was the fixed
    spline), `pos_slope` now defaults to `False`, and the figure switch is
    `plot=` (`make_plot=` is deprecated). The pith-offset convention also changed
    — see the note below.

!!! note "Pith-offset convention (changed in 0.8.0)"

    dplPy's `pith_offset` is a **true offset**: the number of rings *missing*
    between the pith and the first measured ring, so `0` means the pith is
    present. This is exactly ARSTAN's `nshft`, so an ARSTAN pith-offset file can
    be used directly. **dplR** instead stores the cambial age of the first ring
    (`1` = pith present), so subtract 1 from a dplR pith-offset table before
    using it here. With no `po` given, dplPy assumes `pith_offset = 0` (pith
    present) for every series.

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

### How dplPy's RCS compares to ARSTAN

dplPy's `rcs()` matches ARSTAN single-curve RCS in its core: it aligns series by
cambial age from the pith offsets, builds a robust (biweight) mean-by-age regional
curve, and detrends each series by the slice of that curve at its own cambial ages,
as ratios by default. A few differences are worth knowing:

- **Regional-curve shape.** Like ARSTAN and CRUST, dplPy chooses the regional
  curve from a menu via `fit=`: an age-dependent spline (`"AgeDepSpline"`, the
  default), a fixed smoothing spline (`"Spline"`), a parametric growth curve
  (`"ModNegExp"`, `"NegExp"`, `"Hugershoff"`, `"GeneralExp"`, `"LinearAny"`,
  `"LinearNegative"`, `"Mean"`), or `None` for the unsmoothed age-aligned mean.
  This mirrors ARSTAN's curve-fit menu — whose RCS default is likewise the
  age-dependent spline with a non-increasing end — and CRUST's `rdt` options.
- **Difference indices.** With `method="difference"`, dplPy returns residual
  indices centred on 0. ARSTAN instead rescales residual indices onto the ratio index's
  mean and standard deviation (centred on ~1). Set `rescale_difference=True` to
  reproduce that ARSTAN `calind` behavior per series; it is off by default, so
  residuals stay centred on 0.
- **Non-positive curve.** If the regional curve dips to or below zero, ARSTAN
  halts; dplPy issues a warning and continues.
- **Pith offsets** follow the +1 convention noted above.

Signal-free standardization iterates detrending and chronology-building to reduce
the trend-distortion (end-effect) bias of ordinary detrending:

```python
rwi = dpl.ssf(rwl)
```

It repeats until the high-frequency chronology stops changing (median absolute
difference below `mad_threshold`). `preset="crust"` selects the CRUST variant.

## `mrcs`: multiple-curve RCS

A single regional curve assumes every tree shares one age->growth expectation.
Because faster-growing trees tend to be younger ("grow fast, die young"), one
average curve injects a slope bias when the sample's growth-rate or age mix
changes over calendar time. `mrcs` sorts the trees by relative growth rate,
splits them into a few equal-count classes, and standardizes each tree by its own
class curve -- a port of CRUST's multiple-RCS option (Melvin & Briffa 2014).

```python
rwi = dpl.mrcs(rwl, po=pith_offsets, n_curves=2)             # two growth classes
rwi = dpl.mrcs(rwl, po=pith_offsets, n_curves=3, sort="age")
```

Each class curve is built by `rcs`, so `mrcs` shares its curve vocabulary
(`fit`, `method`, `period`, `preset`, ...) and `mrcs(..., n_curves=1)` is
identical to `rcs`. Key options:

- `n_curves` (default `2`) -- number of regional curves. Most of the correction
  (and most of the low-frequency variance loss) appears already at two curves.
- `sort` (default `"growth"`) -- how trees are ranked before the split:
  `"growth"` (CRUST's growth-rate ranking), `"age"`, `"diameter"`, or `"none"`.
- `reset_means` (default `True`) -- multiple RCS removes some genuine
  low-frequency variance (on timescales beyond the mean tree length). With
  `reset_means` each tree's indices are rescaled back to the mean they would have
  had under one-curve RCS, reinstating that variance while keeping the slope
  correction (CRUST's "mean single"; requires `method="ratio"`).

Use `mrcs` with well-replicated collections -- CRUST's rule of thumb is at least
40 trees per curve, and `mrcs` warns below that. Signal-free multiple-curve RCS
(`sfmrcs`) is planned as a separate function.

## `ads` and `powt`

- **`ads`** fits an age-dependent spline to a single series — a spline whose
  stiffness increases with cambial age. It is also available inside `detrend`
  via `fit="AgeDepSpline"`.
- **`powt`** applies a power transformation (Cook & Peters) as an alternative to
  ratio detrending for variance stabilization. `plot=True` draws an ARSTAN-style
  "data transform statistics" diagnostic — per-series skew and spread-vs-level
  correlation, before and after the transform — and `return_stats=True` returns
  those numbers as a table, so you can check the transform actually made the
  series symmetric and decoupled variance from level.

## Choosing a method

Spline and `ModNegExp` are the everyday choices for site chronologies. Reach for
`rcs` when you need to retain long-timescale climate signal, `sfrcs` when you want
RCS *and* the signal-free correction for trend distortion under an uneven age
structure, `ssf` when end-effect bias matters in curve-by-curve detrending, and
`powt`/`difference` detrending when heteroscedastic variance is the concern. See the
[Detrending & standardization reference](../reference/detrending.md) for every
parameter.
