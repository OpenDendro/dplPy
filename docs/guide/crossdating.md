# Crossdating & COFECHA

Crossdating checks that every ring is assigned to its correct calendar year by
correlating each series, segment by segment, against a master built from the
other series. dplPy offers a **dplR-faithful** `xdate`, a **COFECHA emulation**
(`xdate(preset="COFECHA")`), a COFECHA-style batch **report**, and
floating-series **dating**.

## `xdate`: segment correlations and flags

```python
import dplpy as dpl

rwl = dpl.readers("ca533.rwl", header=True)
result = dpl.xdate(rwl)
result["seg_corr"]     # per-series, per-segment correlations
result["flags"]        # A/B flags per series
```

`xdate` returns a dictionary: `seg_corr` (a series × segment correlation table),
`flags`, and `bins`. By default it follows dplR's `corr.rwl.seg`:

- **Prewhitening** with a Yule-Walker AR model (`prewhiten=True`).
- **Spearman** correlation (`corr="spearman"`).
- A **biweight leave-one-out master** — each series is compared against the mean
  of the *others*, so a misdated series cannot prop up its own correlation.
- Overlapping **segments** of `slide_period` years (default 50), stepped by half
  that, with bins floored to `bin_floor` (default 100).

### Reading the flags

- **A flag** — the segment's correlation with the master is not significant at
  `p_val` (default 0.05). This reproduces dplR exactly: a segment is flagged when
  its correlation fails to clear the critical value.
- **B flag** — the correlation is *higher at a non-dated lag* than at the dated
  position, hinting at a possible dating shift. dplPy's B flag follows COFECHA
  (it fires whenever the best match is at a non-zero lag, with no extra margin);
  dplR's `corr.rwl.seg` has no B flag, so this is a deliberate, documented
  addition that is on in both the default and COFECHA modes.

Set `make_plot=True` for a dplR-style crossdating plot, or use `series_corr` to
drill into one series.

## COFECHA emulation

Passing `preset="COFECHA"` switches `xdate` to emulate the COFECHA program
instead of dplR:

```python
result = dpl.xdate(rwl, preset="COFECHA")
```

The preset changes the crossdating machinery to match COFECHA:

- **Burg** (maximum-entropy) AR prewhitening,
- **spline variance stabilization** of the series,
- an **arithmetic, z-scored** leave-one-out master,
- **Pearson** correlation,
- COFECHA **segment anchoring** and **t-based critical values**, and
- **"omit absent rings"** handling.

The dplR-faithful default is unchanged; the preset is an opt-in second mode.

## `xdate_report`: COFECHA-style batch reports

`xdate_report` runs one or more ITRDB files through crossdating and writes a
formatted, COFECHA-style `.txt` report with per-series and per-segment
correlations and a length-weighted summary:

```python
report = dpl.xdate_report("ca533.rwl", preset="COFECHA")
```

It accepts a single file or a list, takes the same `slide_period` / `bin_floor` /
`p_val` settings as `xdate`, and offers both a dplR-faithful and a
`preset="COFECHA"` mode. Set `write=False` to get the report text back without
writing files.

## Diagnostics for one series

`series_corr` reports the correlation of a single named series against the master
segment by segment (useful when `xdate` flags a series and you want to see
where), and `interseries_corr` returns the overall interseries correlation of the
collection — the same headline number COFECHA reports.

```python
dpl.series_corr(rwl, series_name="CAM011")
dpl.interseries_corr(rwl)
```

## Dating a floating series

`xdate_floater` places an *undated* (floating) series — a beam, a remnant log —
against a dated reference collection by sliding it across every offset and
scoring the fit:

```python
placed = dpl.xdate_floater(reference_rwl, floating_series, series_name="beam1")
placed["best"]      # best-fit calendar placement and its statistics
```

It is based on dplR's `xdate.floater` and extended with the *t* statistic and
additional crossdating statistics of Wilson (2026); the best placement is chosen
by highest *t* (see the [reference](../reference/crossdating.md#xdate_floater)
for the two documented departures from dplR).

## dplR vs COFECHA at a glance

| | Default (`preset=None`) | `preset="COFECHA"` |
|---|---|---|
| Emulates | dplR `corr.rwl.seg` | COFECHA |
| Correlation | Spearman | Pearson |
| Prewhitening | Yule-Walker AR | Burg (max-entropy) AR |
| Master | biweight, leave-one-out | arithmetic, z-scored, leave-one-out |
| Critical value | p-value (`p_val`) | COFECHA t-based table |
| A flag | yes (matches dplR) | yes |
| B flag (lag shift) | yes (COFECHA-style) | yes |

See the [Crossdating reference](../reference/crossdating.md) for every parameter.
