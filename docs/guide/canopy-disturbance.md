# Canopy disturbance detrending

A tree that loses a competitor overhead grows faster for decades afterwards. That
release is real biology, but it is not climate, and when it survives into a
standardized chronology it masquerades as a low-frequency climate signal. Canopy
disturbance detrending finds those growth releases and removes them, so that what
is left tracks climate more faithfully — and, as a by-product, it hands you the
disturbance history itself, which is often the thing a forest ecologist wanted all
along.

dplPy implements the method of Druckenbrod, Cook, Pederson & Martin-Benito (2024),
a direct port of Dan Druckenbrod's MATLAB that is validated number-for-number
against it (see [Fidelity & departures](../background/fidelity.md)). The surface is
three functions: `releases` to detect, `disturbance_detrend` to standardize, and
`disturbance_plot` to look.

## How the method works

For each series, in order:

1. **Detect releases by radial-growth averaging** (Lorimer & Frelich 1989). A
   trailing running mean of ring width is compared window-to-window: the *percent
   growth change* at a year is the increase from the preceding window's mean to the
   subsequent window's mean. A run of years whose percent growth change clears a
   threshold is a release, pinned to its year of maximum change.
2. **Power-transform** the series to stabilize its variance (Cook & Peters 1997),
   so the spline fits that follow are not dominated by the high-growth years.
3. **Remove each release** in turn, newest to oldest, by fitting a short
   age-dependent spline (Melvin et al. 2007) from the release year onward and
   subtracting it — while the release sustains elevated growth. A release that is
   only transient is removed just up to the point where growth returns to its
   pre-release level.
4. **Age-detrend** the disturbance-free series with a second, stiffer
   age-dependent spline.
5. **Re-express** in ring-width units. The climate index is the ratio of the
   disturbance-standardized series to the age trend.

## Detecting releases

`releases` runs step 1 on its own and returns a tidy table, one row per event:

```python
import dplpy as dpl

rwl = dpl.readers("mo024.rwl")        # Stahle's Missouri post oak, 61 series
rel = dpl.releases(rwl)
#   series  year  pct_growth_change
#   DEM01A  1915         216.16
#   DEM03A  1809          64.25
#   ...                           (series-then-year order)
```

The disturbance history of the stand falls straight out of this table — count the
events per decade and you have the release chronology that the summary figure
draws. Detection runs on **raw** ring widths; do not power-transform first, because
the threshold is calibrated on raw growth.

Two parameters set the sensitivity:

```python
dpl.releases(rwl, threshold=50, window_length=15)   # defaults (recommended)
dpl.releases(rwl, threshold=25, window_length=10)    # more sensitive, more events
```

Druckenbrod et al. recommend 50 % over 15 years over the more permissive 25 % /
10 years; the method needs at least `2 * window_length` rings in a series.

## Standardizing

`disturbance_detrend` runs the whole pipeline and, by default, returns the
per-series climate index — the same shape and role as [`detrend`](detrending-standardization.md)
output, so it drops straight into a chronology:

```python
idx = dpl.disturbance_detrend(rwl)          # DataFrame: years x series
crn = dpl.chron(idx)                         # standard chronology
```

A single series in returns a single series out. For the full set of intermediate
products — the disturbance-standardized series, the age trend, the residual index,
the release table, and the averaged chronology behind the summary figure — ask for
the dict:

```python
info = dpl.disturbance_detrend(rwl, return_info=True)
info["index"]                    # the climate index (St = DSt / At)
info["disturbance_standardized"] # DSt: growth with the releases removed
info["age_trend"]                # At
info["averages"]                 # mean width, age trend, Dt, index, sample depth
info["releases"]                 # the detection table
```

## Looking at the result

Plotting is built so a many-series call never floods you with figures.
`disturbance_detrend(rwl, plot=True)` draws a **single** summary figure for a whole
chronology — the release histogram with sample depth, the mean widths with the
disturbance removed and the age trend, and the mean index:

```python
dpl.disturbance_detrend(rwl, plot=True)         # one summary figure
```

To inspect individual cores, name them with `disturbance_plot`, which draws the
three-panel per-series view (detection, disturbance removal, index):

```python
dpl.disturbance_plot(rwl, series="DEM01A")              # one core
dpl.disturbance_plot(rwl, series=["DEM01A", "DEM07B"])  # a chosen handful
dpl.disturbance_plot(rwl)                                # the summary again
```

Passing `disturbance_detrend` a single Series with `plot=True` draws that series'
three-panel figure, since there is only one series to show.

## Choices you may need to make

**Spline stiffness.** `release_flex=10` (years) sets how closely each release event
is followed and `age_flex=30` the stiffness of the final age trend. Both are fixed
year-lengths, as in the published method.

**The power transform.** `transform="druckenbrod"` (the default) is the method's
own transform and reproduces the published results. `transform="powt"` substitutes
dplPy's standard Cook & Peters [`powt`](../reference/detrending.md); it departs
materially from the published method and is offered only for workflows that prefer
the house transform. `transform=None` works in raw ring-width units with no
transform. The transform and its inverse are the same engine as `dpl.powt`.

**Absent rings.** A locally absent ring is recorded as a zero — a real year with no
measurable growth, distinct from missing data (`NaN`). Scattered zeros pass through
as zero growth, but a zero in a series that takes the log branch of the power
transform would otherwise produce `NaN`. `recode_zeros=True` recodes zeros to
0.001 before processing (the same convention as [`ssf`](../reference/detrending.md);
the input is never modified); left `False`, a series containing zeros is detected
as-is and a warning names it.

**The denominator guard.** `min_mean=0.01` (mm) keeps a near-zero baseline — a
window dominated by absent or recoded rings — from inflating the percent growth
change into a spurious release. The default touches only such pathological windows;
`min_mean=0` reproduces the unguarded method exactly.

## Coming from TRADER or dplR

dplR has no canopy-disturbance detrending. The R package **TRADER** implements the
*detection* side (several boundary-line and radial-growth-averaging variants);
dplPy's `releases` covers the radial-growth-averaging detector, and
`disturbance_detrend` goes further, removing the releases and returning a climate
index — the detrending step TRADER does not do.

## Fidelity

The detector reproduces Druckenbrod's MATLAB release years exactly on his `mo024`
test set (94 releases across 52 of 61 series), and the full pipeline reproduces the
per-series disturbance series, age trend, index and residual to float64 round-off.
The age-dependent spline reuses dplPy's [`ads`](../reference/detrending.md), which
matches Druckenbrod's `ADSmelvin` to the same tolerance. See
[Fidelity & departures](../background/fidelity.md) for the package-wide principle
and [References](../background/references.md) for the works behind the method.
