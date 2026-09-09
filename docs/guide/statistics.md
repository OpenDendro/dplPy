# Statistics & agreement

Beyond building a chronology, dplPy provides the descriptive statistics, signal-
strength measures, and agreement tests used to characterize a collection and
judge whether it carries a common, datable signal.

## Descriptive statistics: `stats`

`stats` reports per-series summary statistics — span, mean, median, standard
deviation, skew, kurtosis, the Gini coefficient, and first-order autocorrelation:

```python
import dplpy as dpl

rwl = dpl.readers("ca533.rwl", header=True)
dpl.stats(rwl).head(3)
```

```text
   series  first  last  year   mean  median  stdev   skew  kurtosis   gini    ar1
1  CAM011   1530  1983   454  0.440    0.40  0.222  1.029     1.102  0.273  0.696
2  CAM021   1433  1983   551  0.424    0.40  0.185  0.946     1.110  0.237  0.701
3  CAM031   1356  1983   628  0.349    0.29  0.214  0.690    -0.366  0.341  0.808
```

`ar1` here is the autocorrelation-function coefficient at lag 1 (matching dplR's
`rwl.stats`), not an OLS AR(1) slope.

## Mean sensitivity: `sens1` and `sens2`

Mean sensitivity measures year-to-year variability. `sens1` is the classic
Douglass measure; `sens2` is the trend-robust variant (Biondi & Qeadan 2008).
Both take a single series:

```python
s = rwl["CAM011"].dropna()
dpl.sens1(s)   # 0.3437
dpl.sens2(s)   # 0.2960
```

As in dplR, sensitivity is deliberately kept out of the `stats` table and exposed
as its own functions.

## Signal strength: `rwi_stats`, `sss`

`rwi_stats` computes the population-signal statistics on **detrended** series —
inter-series correlation (rbar), the expressed population signal (EPS), and the
signal-to-noise ratio (SNR). It groups cores by tree using an `ids` table from
`read_ids`:

```python
rwi = dpl.detrend(rwl, fit="Spline")
ids = dpl.read_ids(rwl)
dpl.rwi_stats(rwi, ids)
```

```text
   n_cores  n_trees   n  n_tot  ...  rbar_eff    eps     snr
0       34       34  34    523  ...     0.423  0.961  24.875
```

`rwi_stats_running` computes the same statistics in a moving window to see how
signal strength changes through time. `sss` (subsample signal strength) estimates
how well a reduced sample reproduces the full-collection signal — useful for
deciding how far back a chronology remains reliable.

## Agreement: `glk` and `sgc`

`glk` (Gleichläufigkeit) and `sgc` (synchronous growth changes) measure the
proportion of intervals in which two series move in the same direction — a
classic non-parametric agreement test. Both return a dictionary of pairwise
values and their significance:

```python
dpl.glk(rwl, overlap=50)
dpl.sgc(rwl, overlap=50)
```

The p-values follow dplR's formulation (a p-value can exceed 1 when the statistic
is below 0.5).

## Tree-level and interval helpers

- **`tree_mean`** averages cores to the tree level first (using an `ids` table),
  so multi-core trees are not over-weighted in downstream statistics.
- **`common_interval`** finds the interval or subset of series that maximizes
  overlap (by years, by series, or both) — a faithful port of dplR's
  `common.interval`, handy before computing statistics that need complete data.

## Basal area increment: `bai_out` and `bai_in`

`bai_out` and `bai_in` convert ring widths to basal area increment, working
inward from the outermost ring (`bai_out`) or outward from the pith (`bai_in`,
which can take a pith-to-first-ring distance):

```python
dpl.bai_out(rwl)
dpl.bai_in(rwl, d2pith=pith_distances)
```

See the [Statistics & indices reference](../reference/statistics.md) for the full
parameter lists.
