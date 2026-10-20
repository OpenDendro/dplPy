# Climate–growth response

`dpl.seascorr` identifies the seasonal climate signal in a tree-ring chronology:
it correlates the chronology against a *primary* and a *secondary* monthly
climate variable across many candidate seasons and flags which relationships are
significant. It is a faithful port of SEASCORR (Meko, Touchan & Anchukaitis
2011, *Computers & Geosciences* 37, 1234–1241), the first step in moving from a
dated chronology toward a climate reconstruction.

The chronology is correlated (simple Pearson) with the primary variable and
(partial, controlling for the primary) with the secondary variable, for every
combination of a set of season-lengths and 14 ending months. Significance is
assessed by *exact simulation*: the chronology is simulated many times in a way
that preserves its spectrum, and each observed correlation is compared with the
distribution of correlations from the surrogates.

## Basic usage

```python
import dplpy as dpl

crn = dpl.chron(dpl.detrend(dpl.readers("site.rwl")))   # a chronology
result = dpl.seascorr(crn, primary=precip, secondary=temperature)
result.table          # one row per season: correlations, p-values, significance
```

`crn` is a year-indexed `Series`, or a `DataFrame` with `column=` naming the
index series. `primary` and `secondary` are each **one** monthly climate
variable. The default pairing follows the paper — precipitation primary
(summed over a season), temperature secondary (averaged):

```python
result = dpl.seascorr(
    crn, primary=precip, secondary=temperature,
    end_month=9,                      # last month of the growth year
    season_lengths=(1, 3, 6, 12),     # months; the first must be 1
    primary_agg="sum", secondary_agg="mean",
    nsim=1000,                        # exact simulations; >= 1000
    seed=42,                          # reproducible significance
)
```

To put temperature first (simple correlation on temperature, partial on
precipitation), just swap the arguments — there is no role flag:

```python
result = dpl.seascorr(crn, primary=temperature, secondary=precip,
                      primary_agg="mean", secondary_agg="sum")
```

## Climate input formats

Each climate variable may be supplied in whichever shape you already have; all
three normalize internally to the same monthly series:

```python
# wide: a year index/column + 12 month columns (a station CSV)
dpl.seascorr(crn, primary=wide_df, secondary=temp_df)

# tidy long: columns year, month, value
dpl.seascorr(crn, primary=tidy_df, secondary=temp_df)

# datetime-indexed monthly series (e.g. a grid-point extraction)
pre = ds["pre"].sel(lat=36.1, lon=8.5, method="nearest").to_series()
dpl.seascorr(crn, primary=pre, secondary=tmp)
```

Units are your responsibility (scale e.g. tenths of mm to mm before calling);
`primary_agg`/`secondary_agg` then set whether a season is a sum or a mean.

## The result

`seascorr` returns a `SeascorrResult`. Its main payload is `result.table`, a tidy
DataFrame with one row per season:

| Column | Meaning |
| --- | --- |
| `season_length`, `end_month`, `label` | the season (e.g. `Mar–May`, `Oct*` for a prior-year ending month) |
| `r_primary`, `p_primary` | simple correlation with the primary variable and its empirical non-exceedance probability |
| `r_partial`, `p_partial` | partial correlation with the secondary variable (controlling for the primary) and its probability |
| `r_clim` | Pearson r between the primary and secondary variable |
| `sig05_*`, `sig01_*` | two-sided significance flags at α = 0.05 / 0.01 |

Also on the result: `result.highest` (the strongest primary-variable season per
season-length), `result.acf` (lag 1–5 autocorrelation of the chronology), and
`result.period`.

## The bar chart

With `plot=True` (the default) `seascorr` draws the one figure the method is
built around — the two-panel seasonal bar chart (simple correlations on top,
partial on the bottom), significance color-coded. Bar color follows the
*variable*: temperature is red, moisture (precipitation, PDSI, …) blue,
whichever is primary. It is auto-detected from `primary_name`/`secondary_name`,
and can be forced:

```python
res = dpl.seascorr(crn, primary=temperature, secondary=precip,
                   primary_name="Temperature", secondary_name="Precipitation")
res.fig.savefig("seascorr.png", dpi=300)       # fig and axes hang off the result

# force colors, or a black-and-white version for print
dpl.seascorr(crn, primary=p, secondary=t, primary_color="blue",
             secondary_color="red")
dpl.seascorr(crn, primary=p, secondary=t, plot=True, ... )   # color=False in the plot call for B/W
```

Each panel's y-axis is fixed at ±0.3 and expands only to accommodate larger
correlations, so a weak, non-significant partial panel still reads on a stable
scale.

## Temporal stability

Pass an early and a late window as `period` to test whether the strongest
season's correlation is stable between the two sub-periods (Fisher-z difference
of correlations, with effective sample sizes adjusted for autocorrelation after
Dawdy & Matalas 1964):

```python
res = dpl.seascorr(crn, primary=precip, secondary=temp,
                   period=((1903, 1952), (1953, 2002)))
res.stability     # one row per season-length: r_full/early/late, Δz, p_value
```

A single window, `period=(first, last)`, just sets the analysis period; omitting
`period` uses the full overlap of chronology and climate.

## Things to know

- **Per-season significance.** Each of the seasons is tested independently at
  its own α; `seascorr` does not apply a multiple-comparison correction, faithful
  to Meko et al. (2011). Read a lone significant season among many with that in
  mind.
- **Exact simulation.** It is the *chronology* that is simulated (preserving its
  spectrum), so the null is "a tree-ring series with this spectrum, unrelated to
  climate." `nsim` must be at least 1000 so the tail probabilities can resolve
  α = 0.01. Set `seed=` for reproducible p-values.
- **Partial correlations** behave very slightly liberally over heavily
  overlapping seasons; treat marginal partial-correlation significance cautiously.
- **Data requirements.** The chronology must be gap-free over the analysis window
  and at least 31 years long (fill interior gaps with
  [`fill_internal`](internal-gaps.md) first). Each climate variable needs
  complete monthly coverage over the years it contributes; because long seasons
  reach into the prior year, the usable window starts about two years after the
  first year of complete climate data.
- **Non-Gaussian chronologies** still run, with a warning — the exact-simulation
  method assumes an approximately Gaussian series.

See the [Climate response reference](../reference/climate.md) for the full
`seascorr` and `SeascorrResult` signatures.
