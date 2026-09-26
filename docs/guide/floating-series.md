# Dating a floating series

A *floating* series is undated wood — a beam from a historic building, a remnant
or sub-fossil log — whose rings are measured but not yet tied to calendar years.
`xdate_floater` estimates its calendar placement by sliding it across a dated
reference collection and scoring the fit at every offset.

## The idea

You need two things:

- a **dated reference**, either a whole collection (a `DataFrame` from `readers`,
  indexed by calendar year, from which a master chronology is built by averaging
  the series) or a single, already-built chronology passed as a `pandas.Series` or
  a one-column `DataFrame` — for example a column from `read_crn`; and
- the **floating series** itself (a `pandas.Series` of ring widths; its index
  does not need to be calendar years).

`xdate_floater` builds (or takes) a master from the reference, transforms the
floater the same way, and tests every alignment with at least `min_overlap`
overlapping rings, reporting the correlation and crossdating statistics at each.
The floating series may be longer than the reference.

**Dating against an existing chronology.** A `.crn` read with `read_crn` already
*is* a master, so pass the column you want (e.g. `crn["std"]`) directly as the
reference. For an already-prewhitened/residual chronology (an ARSTAN `res`/`ars`),
also pass `transform="none"` so it is not prewhitened a second time.

## Example

```python
import dplpy as dpl

reference = dpl.readers("reference_site.rwl", header=True)
placed = dpl.xdate_floater(reference, floating_series, series_name="beam1")
```

```text
Best match for 'beam1': 1727 to 1983  (t = 17.83, r = 0.747, eff_df = 254,
p_bonf = 8.52e-44, 1/p = >1 million, IF = >1000, overlap n = 254)
```

If you don't pass `series_name`, it is taken from the series itself (a named
`Series`, or a one-column `DataFrame`'s column header). To date against a published
chronology rather than a collection, hand it the column you want:

```python
crn = dpl.read_crn("reference_site.crn")
placed = dpl.xdate_floater(crn["std"], floating_series, series_name="beam1")
```

The return value is a dictionary:

- `placed["best"]` — the top-scoring placement: `min_year`, `max_year`, `r`, `t`,
  `eff_df`, `p_bonf`, `one_over_p`, `isolation_factor`, and `n`.
- `placed["floater_cor_stats"]` — every tested offset, highest-*t* first, so you
  can inspect the runners-up.
- with `return_rwl=True`, also `placed["placed"]` (the floater at its best-fit
  years) and `placed["combined"]` (floater merged with the reference).

## Reading the statistics

| Statistic | What it tells you |
|---|---|
| `r` | The sliding correlation between floater and master at that offset. |
| `t` | The Baillie & Pilcher (1973) *t* statistic, `r·√(df−2)/√(1−r²)`. Higher is stronger; values ≥ ~3.5 are the classic acceptance threshold. |
| `eff_df` | Effective degrees of freedom, reduced from the raw overlap for autocorrelation (Wigley et al. 1987). |
| `p_bonf` | One-tailed p-value of *t*, Bonferroni-corrected for the many offsets tested. |
| `isolation_factor` (IF) | How many times larger the runner-up's p-value is than the best offset's. A large IF means the date stands out cleanly from the alternatives. |

The **best placement is chosen by the highest *t***, which accounts for the
degrees of freedom at each overlap. A confident date has a high *t*, a tiny
`p_bonf`, and a large isolation factor — as in the example above.

## Useful arguments

- `min_overlap` (default 50) — the minimum number of overlapping rings an
  alignment must have to be scored.
- `transform` (default `"pw"`) and `prewhiten` — how the series are detrended and
  prewhitened before correlation.
- `corr` (default `"spearman"`) — the correlation method.
- `series_name` — a label for the floater used in the outputs and plot title; when
  omitted it is taken from a named series or one-column frame, else `"Unknown"`.
- `make_plot=True` — draw the dating figure: the sliding *t* and the
  Bonferroni-adjusted p-value against calendar year, the distribution of all *t*
  values with the best marked, and an overlay of the reference chronology against
  the best-placed floater over the dated overlap.

## When the best date is impossible

dplPy also scores placements that run past the end of the reference into future
calendar years — this enriches the null distribution of *t* values and acts as a
sanity check. If the *best* placement lands after the present year, though, the
date is impossible and the match is almost certainly spurious, so `xdate_floater`
flags it loudly: `best["date_warning"]` is set to `"future"`, a warning banner is
printed (even with `verbose=False`), and the returned plot carries a red
"best date is in the future" banner. Check the series orientation (oldest →
youngest) and the reference before trusting such a result.

## Relationship to dplR

`xdate_floater` is based on dplR's `xdate.floater` and extends it with the *t*
statistic and additional crossdating statistics of Wilson (2026). It differs from
dplR in two documented ways: the best placement is selected by highest *t* (dplR
uses the highest correlation *r*), and dplPy does not apply dplR's future-year
trim — instead it tests those offsets and *warns* when the winning date is an
impossible future one (see [When the best date is impossible](#when-the-best-date-is-impossible)). See the
[`xdate_floater` reference](../reference/crossdating.md#xdate_floater) and
[Coming from dplR](coming-from-dplr.md).
