# Building chronologies

A chronology averages the detrended (RWI) series into a single site record. dplPy
provides a standard mean-value chronology, an AR-modelled variant, and a
variance-stabilized variant.

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
  chronology. When on, `max_lag`, `aic`, `ar_method` (`"yw"` Yule-Walker, matching
  dplR's default, or `"ols"`), and `first_aic_min` control the AR model.

## `chron_ars`: AR-based chronology

`chron_ars` builds an ARSTAN-style chronology, pooling AR structure across series:

```python
crn = dpl.chron_ars(rwi, max_lag=10, first_aic_min=True)
```

It returns the standard, residual, and ARSTAN chronologies with unified column
names. `prewhiten_method="ar.yw"` uses Yule-Walker AR to match dplR.

## `chron_stabilized`: variance-stabilized chronology

Sample depth changes through time, which changes chronology variance. `chron_stabilized`
adjusts for this using the running inter-series correlation and effective sample
size:

```python
crn = dpl.chron_stabilized(rwi, win_length=50, min_seg_ratio=1/3)
```

- `win_length` — the moving window (years) over which rbar and effective sample
  size are computed.
- `min_seg_ratio` — the minimum fraction of the window a series must cover to
  count toward rbar. It defaults to exactly `1/3`, matching dplR's
  `chron.stabilized` (which drops overlaps below `win_length/3`).
- `method` — `"running_rbar"` (Frank et al. 2006), `"briffa"` (ARSTAN's Briffa
  method), or `"spline"` (ad-hoc spline variance stabilization).

## Which to use

`chron` is the default site chronology. Use `chron_ars` when you want the
residual/ARSTAN chronologies and pooled AR modelling, and `chron_stabilized`
when changing sample depth would otherwise bias the chronology's variance
through time. See the [Chronology reference](../reference/chronology.md) for full
parameter lists.
