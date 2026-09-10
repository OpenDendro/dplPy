# Building chronologies

A chronology averages the detrended (RWI) series into a single site record. dplPy
provides two builders that sit at different points on a simplicity–fidelity
spectrum:

- [`chron`](#chron-the-standard-chronology) — **simple but flexible.** The
  everyday mean-value chronology, with optional per-series prewhitening. This is
  the one most workflows want.
- [`chron_ars`](#chron_ars-the-arstan-chronology) — **an ARSTAN emulator.** Built
  specifically to reproduce ARSTAN's parameters and procedures: a common pooled
  AR order across all series, and the standard, residual, and re-reddened
  "ARSTAN" chronologies.

Both can also variance-stabilize their output in a single call — see
[Variance stabilization](variance-stabilization.md).

## `chron`: the standard chronology

Pass detrended series to `chron`:

```python
import dplpy as dpl

rwl = dpl.readers("ca533.rwl", header=True)
rwi = dpl.detrend(rwl, fit="Spline")
crn = dpl.chron(rwi)
crn.tail(3)
```

```text
           std  samp_depth
Year
1981  1.252329          21
1982  1.362052          21
1983  1.314605          21
```

The result has a `std` (standard chronology) column and a `samp_depth`
(sample-depth) column, indexed by year.

- `biweight` (default `True`) — average with Tukey's biweight robust mean rather
  than an arithmetic mean, reducing the influence of outlier series.
- `prewhiten` (default `False`) — also return a residual (prewhitened)
  chronology in a `res` column. When on, the AR model is fit **per series** at
  each series' own optimal order, controlled by `max_lag`, `aic`, `ar_method`
  (`"yw"` Yule-Walker, matching dplR's default, or `"ols"`), and `first_aic_min`.

`chron` is deliberately lightweight: it reproduces dplR's `chron` and gives you a
clean standard (and optional residual) chronology without committing to ARSTAN's
pooled-AR machinery. When that is all you need, it is the right tool.

## `chron_ars`: the ARSTAN chronology

`chron_ars` is not just a mean-value function with prewhitening bolted on — it is
a mimic of Ed Cook's ARSTAN, and its parameters and procedures follow ARSTAN
rather than dplR's `chron` where the two differ. It returns three chronologies
with unified column names:

```python
ars = dpl.chron_ars(rwi, max_lag=10, first_aic_min=True)
list(ars.columns)
# ['std', 'res', 'ars', 'samp_depth']
```

- **`std`** — the standard chronology (the robust mean of the RWI series).
- **`res`** — the residual chronology: each series is prewhitened with a
  **common pooled** AR(p) model, the prewhitened series are averaged, and that
  mean is prewhitened once more. (This pooled, common-order AR is the key
  difference from `chron(prewhiten=True)`, which uses a per-series optimal
  order.)
- **`ars`** — the ARSTAN chronology: the common red-noise persistence is
  reintroduced ("re-reddened") into the prewhitened signal, keeping the shared
  autoregressive structure while suppressing series-specific noise.

The pooled AR order is chosen from an autocovariance accumulated across all
series and lags (Cook's pooled-AR approach), selected by AIC.
`prewhiten_method="ar.yw"` uses Yule-Walker AR to match dplR.

### `ars_method`: how the ARSTAN chronology is built

There are two orders in which the re-reddening and averaging steps can be
combined, and they do **not** commute when sample depth changes through time:

- **`ars_method="arstan"`** (the default) — `ars = postAR(mean(prewhitened))`:
  average the prewhitened series first, then re-redden that single mean. This
  reproduces Ed Cook's ARSTAN Fortran, which re-reddens the residual chronology
  directly. dplPy defaults to this because `chron_ars` is an ARSTAN emulator.
- **`ars_method="dplr"`** — `ars = mean(postAR(prewhitened))`: re-redden each
  series, then average. This matches dplR's `chron.ars`. Use it for exact dplR
  parity of the `ars` column.

The `std` and `res` chronologies, the pooled AR order, and the prewhitening are
identical either way; only the `ars` column differs.

### `backcast`: keeping the first years

An AR(p) filter has no data for its `p` lagged terms at the start of a series, so
the first `p` residuals would normally be undefined. ARSTAN avoids losing them by
*backcasting* — synthesizing `p` pre-sample values by running the AR model in
reverse (its `bckcst` routine) — at every prewhitening and re-reddening step.
`chron_ars` does the same:

- **`backcast=True`** (the default) — the residual and ARSTAN chronologies are
  **full length**, with no leading `NaN`. This matches ARSTAN, which backcasts
  both the per-series prewhitening and the re-prewhitening of the mean, so a young
  series' first `p` residuals are *included* in the robust mean rather than
  dropped.
- **`backcast=False`** — the first `p` residuals are set to `NaN`, reproducing
  dplR's `chron.ars`.

The backcast values are model-based estimates, so treat the first `p` years as
slightly less certain than the interior. This applies to the `"ar.yw"` path; the
`"arima.CSS-ML"` path already returns full-length residuals.

## Variance-stabilizing a chronology

Both builders accept a `stabilize=` argument (`"rbar"`, `"spline"`, or `"both"`)
that adds a stabilized version of each chronology beside it:

```python
crn = dpl.chron(rwi, prewhiten=True, stabilize="both")
# columns: std, std_vsc, res, res_vsc, samp_depth

ars = dpl.chron_ars(rwi, stabilize="both")
# columns: std, std_vsc, res, res_vsc, ars, ars_vsc, samp_depth
```

You can also stabilize a chronology you already built with `stabilize_chron`, or
use the dplR-parity `chron_stabilized`. The full treatment — the method families,
the running vs constant rbar, and the ARSTAN workflow — is on the
[Variance stabilization](variance-stabilization.md) page.

## Which to use

Use `chron` for a standard site chronology — it is the simplest path and covers
most needs, with `prewhiten=True` when you also want a residual chronology. Reach
for `chron_ars` when you specifically want ARSTAN's behavior: the pooled
common-order AR model and the residual and re-reddened ARSTAN chronologies. Add
`stabilize=` to either when changing sample depth would otherwise bias the
chronology's variance through time, and use `chron_stabilized` when you need exact
dplR `chron.stabilized` parity. See the
[Chronology reference](../reference/chronology.md) for full parameter lists.
