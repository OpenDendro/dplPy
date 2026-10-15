# Handling internal gaps

An *internal gap* is a missing measurement (`NaN`) sitting between a series'
first and last measured ring — a year inside the record where there is no value.
They turn up for ordinary reasons: a ring obscured by rot or a scar, a damaged
core segment, a value dropped in data entry, or a deliberately unmeasured
stretch. This page explains what a gap means in dplPy, why it matters for some
detrending methods and not others, how ARSTAN dealt with it, and — the part that
matters most — how **you** decide what to do about it.

The guiding principle in dplPy is that an internal gap is a modelling decision,
not a detail to be handled silently. dplPy will tell you when a gap could affect
a result and give you the tools to deal with it, but it will never fill your data
behind your back.

## A gap is not a zero

Two things look similar in a ring-width file but mean opposite things:

- A **measured zero** is data. It records a *locally absent ring* — a year the
  tree grew so little (or not at all at that height) that no ring formed. It is
  dated, real, and must be kept.
- An **internal `NaN`** is the *absence* of a measurement — the value is unknown.

dplPy keeps these strictly separate. `dpl.readers()` reads a true interior gap as
`NaN` and a measured zero as `0.0`; it never silently turns one into the other.
(Older versions of dplR once substituted zero for missing — dplR no longer does
this, and dplPy never did.) A measured zero is carried through detrending as a
small positive index; a `NaN` is treated as missing everywhere.

## How each fit treats a gap

dplPy fits every detrending curve to the series with its `NaN` years **dropped**,
so a gap can never corrupt a fit the way a sentinel value (ARSTAN's historic
`-9.99`) could. What differs between methods is the *axis* the curve is fit
against, and since v0.9.0 every axis is the true one:

| Fit | x-axis | Effect of an interior gap |
|---|---|---|
| `Spline` | the real calendar years | none beyond the missing points; the smoothing spline spans the gap in time |
| `Mean` | none (an average) | none; order-invariant |
| `ModNegExp`, `NegExp`, `ModHugershoff`, `Hugershoff`, `GeneralExp` | true ring position (`year - first_year + 1`) | none beyond the missing points; a gap advances cambial age exactly as the calendar does |
| `LinearAny`, `LinearNegative` | true ring position | none beyond the missing points |
| `AgeDepSpline` | true ring position | the gap years are infilled by carrying the last measured value forward so the age-dependent stiffness stays on the true age axis; the curve is read back only at measured rings |

Before v0.9.0 the position-based fits reconstructed cambial age by counting the
*surviving* rings `1, 2, ..., n`, so every ring after a five-year gap was fit five
years too young, and `dpl.detrend()` warned about it. That compression is gone:
a ring at true cambial age 50 is fit at age 50 whether or not there is a gap
before it, and no warning is needed. The same rule holds in the regional-curve
functions (`rcs`, `sfrcs`, `mrcs`, `sfmrcs`), where a gap advances cambial age in
the aligned matrix exactly as CRUST and dplR do, and the gap years come back as
`NaN` in the indices rather than shifting every later year.

!!! note "dplPy does not fill gaps for you"

    A gap stays `NaN` in the detrended output. If you want a value there, infill
    it deliberately with `dpl.fill_internal()` before detrending (see below) so the
    choice is visible in your workflow. dplPy does not change your data silently.

## How ARSTAN handled it (and why dplPy does not copy that)

ARSTAN filled interior gaps **automatically**, before any curve fitting or
chronology building, and you could not turn it off. Its reader flagged negative
"missing" values, and whenever any series had one it ran its `fillin` routine:
fit each tree a flexible growth curve that glides across the gap, build a robust
mean index across all the overlapping trees (the shared stand signal), and for
each missing year impute a value from that common signal — rescaled to the
individual tree's mean and variability and multiplied back by its growth curve.
The gap came out as a modelled ring-width estimate, and standardization proceeded
on a complete series. (The full mechanism, with Fortran line numbers, is in the
developer note `dev/arstan_gap_and_duplicate_id_2026-10-05.md`.)

ARSTAN did this silently because it had to. A Fortran program of its era had no
good way to hand the problem back to the user, no warnings infrastructure, and no
ecosystem of helper functions — and its curve fitters summed over dense
position-indexed arrays that a raw sentinel would wreck. Automatic infilling was
the reasonable engineering choice *at the time*.

dplPy is in a different world. Python treats `NaN` as a first-class missing value
that is dropped before fitting and skipped in aggregation, so nothing *forces* an
infill. That lets dplPy follow a different philosophy: surface the problem, offer
the options — including a faithful port of Ed Cook's own method — and let the
analyst own the choice. Reproducing ARSTAN's fill is one line away when you want
it; the difference is that you ask for it.

## Step 1 — find the gaps

`dpl.readers()` already flags short interior gaps on import and records every one
on the dataframe:

```python
import dplpy as dpl
rwl = dpl.readers("site.rwl")

rwl.attrs["dplpy_interior_gaps"]   # list of {series, gap years, short/long} findings
```

`dpl.report()` gives a fuller picture, listing both locally absent rings (measured
zeros) and internal `NaN` years, per series and in total:

```python
dpl.report(rwl)
```

## Step 2 — decide, then act

There is no universally correct choice; it depends on the fit you want and how
large the gaps are. The three sensible paths:

### Option A — use a spline (no infill needed)

If a cubic smoothing spline suits your site, just detrend. `Spline` spans the gap
in calendar time, so no infill is required and no warning fires:

```python
rwi = dpl.detrend(rwl, fit="Spline")
crn = dpl.chron(rwi)
```

### Option B — infill the gap deliberately with `fill_internal`

If you specifically want a parametric growth curve (or an age-dependent spline),
infill first so the cambial-age axis is continuous, then detrend. `fill_internal`
only touches *interior* `NaN`; leading/trailing `NaN` and measured zeros are left
alone.

```python
# simple, within-series fills (ports of dplR's fill.internal.NA):
filled = dpl.fill_internal(rwl, fill="Mean")     # flat at the series mean
filled = dpl.fill_internal(rwl, fill="Linear")   # straight-line interpolation
filled = dpl.fill_internal(rwl, fill="Spline")   # cubic-spline interpolation

# Ed Cook's ARSTAN method: impute each gap from the population's common signal,
# scaled to the individual series and modulated by its own growth curve:
filled = dpl.fill_internal(rwl, fill="ARSTAN")

rwi = dpl.detrend(filled, fit="ModNegExp")
crn = dpl.chron(rwi)
```

The simple modes interpolate within each series only and are best for short gaps
where the surrounding rings carry the information. `fill="ARSTAN"` borrows the
shared year-to-year signal from the whole collection, which is the point of
Cook's method: it restores *climate-like* variability across the gap rather than
a smooth interpolation, and is the better choice for longer gaps in a
well-replicated collection. Its behaviour is tunable (`growth_nyrs`, `long_gap`,
`flank`, `stabilize`), defaulting to ARSTAN's 20-year growth spline.

### Option C — accept the distortion

For a short gap under a smooth growth curve, the age compression is often
negligible. You can simply proceed with the parametric fit; the warning is a
record of the decision, not an error.

## A worked example

```python
import dplpy as dpl

rwl = dpl.readers("site.rwl")
dpl.report(rwl)                       # see which series have interior NaN

# You want ModNegExp curves AND a couple of series have multi-decade gaps,
# so infill with the common-signal method first, then detrend and build.
filled = dpl.fill_internal(rwl, fill="ARSTAN")
rwi    = dpl.detrend(filled, fit="ModNegExp", method="ratio")
crn    = dpl.chron(rwi)

# Compare against the no-infill spline path to see how much the choice mattered:
crn_spline = dpl.chron(dpl.detrend(rwl, fit="Spline"))
```

## Choosing, in short

Use `Spline` and you can ignore interior gaps entirely. Choose a parametric growth
curve or an age-dependent spline and you should decide, per the gap sizes in your
collection, whether to infill (and with which method) or accept a small age-axis
distortion. For short gaps, `fill="Spline"`/`"Linear"` or simply proceeding is
usually fine; for longer gaps in a replicated collection, `fill="ARSTAN"` restores
realistic variability. The one thing dplPy will not do is decide for you.

See also the [Detrending & standardization](detrending-standardization.md) guide
and the [`fill_internal` reference](../reference/utilities.md).
