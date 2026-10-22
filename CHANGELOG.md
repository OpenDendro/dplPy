# Changelog

All notable changes to dplPy are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and dplPy aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This changelog was started at the `v0.6.0` tag as the project moves toward a 1.0
release; the history of earlier releases (`v0.1.5`–`v0.6.0`) lives in the git tags
and commit log.

## [Unreleased]

### Added
- **`dpl.read_pos` reads CooRecorder `.pos` files** (a single file or a directory
  tree) directly into dplPy, skipping the lossy CDendro `.rwl` export. It returns a
  `PosCollection` with the whole-ring widths (`.rwl()`, full precision, mm),
  earlywood/latewood (`.earlywood()`/`.latewood()`), per-series attributes
  (distance to pith, radius, outer/inner year, comment, version), and the raw
  coordinates for QC. `dpl.readers(path, format="pos")` and `.pos` sniffing return
  just the whole-ring frame. Requires CooRecorder 7.8+ (pith coordinates); gaps are
  measured around (per CRUST/modendro), multipoint via-points are excluded, and
  undated files get relative years. Modeled on `read_pos()` in C. Maher's modendro
  (MIT).
- **`dpl.yrs_to_pith`** estimates a pith offset (rings to pith) from a distance to
  pith and the innermost ring widths (`method="dist"` or `"rings"`), producing a
  pith-offset table for `rcs`/`po_to_wc`. Pairs with `read_pos`. Ports modendro's
  `yrs_to_pith()`.
- **`dpl.seascorr` identifies the seasonal climate signal in a chronology** — a
  faithful port of SEASCORR (Meko, Touchan & Anchukaitis 2011). It correlates the
  chronology (simple Pearson) with a primary monthly climate variable and (partial,
  controlling for the primary) with a secondary variable, across season-lengths ×
  14 ending months, with significance from exact spectral simulation of the
  chronology (Percival & Constantine 2006) and a two-period stability test
  (Fisher-z, Dawdy–Matalas effective sample size). Climate is accepted as wide,
  tidy, or datetime-indexed monthly series; `plot=True` draws the two-panel
  seasonal bar chart, colored by variable (temperature red, moisture blue).
  Verified against the authors' MATLAB to the limit of its printed precision. Lives
  in the new `dplpy.climate` module; also exported as `dpl.seascorr`.
- **`dpl.pcreg` reconstructs climate from a tree-ring network** by orthogonal
  spatial / principal-components regression — a faithful port of Ed Cook's OSR
  FORTRAN (the method of Cook, Briffa & Jones 1994). Predictors are screened against
  the target over the calibration window, reduced to principal components (first-m,
  eigenvalue-1, cumulative-percent, or Guiot PVP truncation), and the target is
  regressed on the PCs that pass a t-test; the fit is back-transformed to
  per-predictor weights, a reconstruction is produced over the full predictor
  period, and it is verified against withheld data (Pearson r, RE, CE, on both the
  undifferenced and 1st-differenced series). Optional AR prewhitening reuses
  dplPy's own `ar_func` (Burg = Cook's maximum-entropy `memcof`). Lead/lag
  predictor masks (`lag=`, Cook's `lagit`) expand each predictor into lagged/led
  copies before screening, given either as integer offsets (`(0, 1)`) or an
  11-char mask (`"00000110000"`). Leave-k-out cross-validation (`cv=`, `cv=1` for
  leave-one-out) refits the whole pipeline on each held-out fold and reports the
  out-of-sample skill (`cv_r`, `cv_RE`, `cv_CE`), exposing overfitting the
  in-sample R² hides. Validated number-for-number against the compiled FORTRAN on
  a single-target fixture (screened predictors, PC count, regression order,
  calibration R², all verification statistics, and the lag expansion all exact;
  reconstruction to 5e-4). dplPy deliberately does not fabricate the FORTRAN's
  common-period edge padding, so a predictor must genuinely cover a lagged window
  to enter it; any predictor dropped for insufficient coverage is reported in a
  warning and kept on `result.dropped_coverage`, rather than silently padded.
  Canonical regression, robust/biweight correlations, and the
  Southern-Hemisphere auto-shift are planned for a later phase. Exported as
  `dpl.pcreg`, returning a `PcregResult`.

### Fixed
- **`ssf(preset="crust")` no longer breaks on interior absent rings.** A single
  interior zero-width ring could raise `'NA's are not allowed in the middle of
  the series`, because a legitimate absent ring was being turned into an illegal
  interior `NaN` during the signal-free rescale. CRUST now keeps absent rings
  (the curve fit recodes them to a small positive width), so it standardises the
  sensitive, many-absent-ring collections it is meant for (e.g. co021) with every
  series retained. Reported by B. Negmetzhanov.
- **`ssf` messages are honest about the method and the remedy.** In
  `method="difference"` mode the non-positive-curve message no longer talks about
  "dividing by" a near-zero chronology (difference subtracts, and the value quoted
  could be negative); it now describes the additive case. The guidance points to
  `preset="crust"` for data with many absent rings and only suggests
  `recode_zeros` for literal zero rings, which it recodes -- `recode_zeros` does
  not help a chronology that is low but positive in a severe pointer year.

- **`ssf`'s non-positive-curve error names the cause and the remedy.** When a
  signal-free curve goes to zero or below, ssf still stops (as dplR does), but the
  message now names the offending series and the near-zero pointer-year chronology
  driving it, and directs the user to `preset="crust"` (which floors the curve and
  standardises such data with every series kept). This replaces dropping series:
  the condition reflects the whole collection at a severe pointer year rather than
  a few anomalies (on co021, dropping would cascade to nearly all series), so no
  arbitrary set-aside threshold is applied.

## [0.9.0] - 2026-10-15

This release is the result of a pre-release audit of the whole library (about
95 items) ahead of the move toward v1.0.0. Many of the entries below change numbers you
may have seen from earlier versions; where they do, the old behaviour was wrong,
undocumented, or inconsistent between functions, and the entry says what moved.
Renamed parameters all keep their old names as deprecated aliases for now.

### Changed
- **The ARSTAN chronology (`chron_ars`, `ars_method="arstan"`) now follows ARSTAN
  exactly.** The `ars` column re-reddens the residual chronology itself (the mean
  of the prewhitened series, prewhitened once more at the pooled order) with the
  pooled AR coefficients, which is what Ed Cook's Fortran does (`albino1` then
  `commie1`) and what the development version of dplR's `chron.ars` has done
  since October 2026. Previously dplPy re-reddened the mean of the prewhitened
  series without the second whitening. `ars_method="dplr"` still reproduces
  released dplR up to 1.8.0 (re-redden each series, then average). On ca533 `ars`
  changes by up to 0.077 after the start-up years and its lag-1 autocorrelation
  moves from 0.48 to 0.51. Prewhitening and re-reddening initials now follow
  ARSTAN's `bckcst` (backcast from the start of each series) rather than dplR's
  `postAR`; only the first few years of each series' contribution differ.
- **One prewhitening procedure across the package.** `chron(prewhiten=True)`,
  `chron_ars`, `rwi_stats(prewhiten=True)`, `xdate`, `series_corr`,
  `interseries_corr`, `xdate_floater`, `xdate_undated`, `ar_func` and `autoreg`
  all fit the AR model the same way: Yule-Walker, maximum order 10, and the
  **first local AIC minimum** (ARSTAN's rule). Before, `prewhiten=True` resolved
  to five different procedures (OLS with `max_lag=5`, global AIC minimum, and so
  on) depending on which function you called. `ar_func`/`autoreg` default to
  `method="yw"`, `ar_max=10`, `first_aic_min=True`. Pass `first_aic_min=False`
  for R's `ar()`/dplR global-minimum rule, and `ar_max=None, first_aic_min=False`
  for dplR's exact crossdating numbers. Burg is available everywhere as the
  ARSTAN/COFECHA estimator (`chron_ars(prewhiten_method="burg")`, the `xdate`
  COFECHA preset). When the AIC never turns up within the ceiling, `autoreg` and
  `chron_ars` raise a clear error instead of silently taking the maximum order
  (the crossdating functions fall back to the minimum within the ceiling).
- **Crossdating lags follow COFECHA everywhere: a negative lag means a missing
  ring in the series.** `xdate` and `check` already did this; `series_corr` did
  the opposite (its lag table slid the series and its `ccf` defaulted to
  `series_x=True`). `series_corr` now defaults to `series_x=False` and shares
  `xdate`'s lag scan, so `xdate`, `check`, `series_corr["lag_table"]` and
  `series_corr["ccf"]` all report the same sign. `series_corr`'s default lag
  search is now 10 years (was 5), like the rest of the package.
- **Parameter names are unified** (old names remain as deprecated aliases that
  warn): window length is `window_length` everywhere (`xdate`/`xdate_report`
  `slide_period`, `series_corr` `seg_length`, `chron_stabilized`/`stabilize_chron`
  `win_length`; `rwi_stats_running` now defaults to 50); the AR order cap is
  `ar_max` (`chron`/`chron_ars`/`ar_func`/`autoreg` `max_lag`); the crossdating
  lag search is `lag` (`check` `lag_max`); the spline wavelength is
  `spline_period` (`detrend`/`rcs`/`mrcs` `period`, `ssf` `nyrs`, `xdate`
  `stabilize_period`, `check`/`chron_stabilized` `spline_nyrs`), with 32 years
  for crossdating high-pass filtering and the 67 % spline for stabilizing
  chronologies; `normalize_for_crossdating(method=)` is `estimator=`; `ssf` uses
  `fit=`/`method=`/`tol=` like `detrend` (`method=<curve>`, `difference=`,
  `mad_threshold=` deprecated); `sfrcs`/`sfmrcs` take a single `biweight=`
  (`biweight_curve`/`biweight_crn` deprecated). Option strings (`fit`, `method`,
  `preset`, `corr`, `format`, ...) are case-insensitive, and `corr` defaults to
  the spelling `"spearman"`.
- **Defaults aligned across functions.** `pos_slope=False` everywhere (now also
  `ads`); `first_aic_min=True` everywhere (`chron` was False);
  `chron_stabilized`'s spline default is the 67 % spline (was 50 %), matching
  `stabilize_chron`; `detrend(plot=False)` is the default, and `plot=True` or
  `plot="sequential"` (one figure at a time) draws; `chron(ci=True)` uses `R=100`
  replicates (dplR's default; was 1000).
- **Return shapes unified.** `writers()` returns the written path and prints
  nothing; `report()` returns the per-series statistics table it prints;
  `interseries_corr` returns a dict `{"mean", "table", "n_series", "too_short"}`
  with the per-series table (column `rho`, rows in input order) instead of a
  tuple; `series_corr["overall"]` is a DataFrame with `rho`/`p_val` like `xdate`
  and `series_corr` also returns the detrended `rwi`; `xdate` flag lags are
  numeric; `stabilize_chron(return_info=True)` returns a dict with the
  chronology under `"vsc"`; `chron_stabilized`'s column is `running_rbar`; an
  unnamed year index is written as `Year`. Every returned frame carries
  `attrs["kind"]` (`rwl`, `rwi`, `crn`, `bai`, `rwl_powt`) and keeps the reader's
  provenance attrs where the frame is still measurements.
- **`rwi_stats_running(ids=...)` is labelled honestly** as ARSTAN-style running
  effective replication (cores present in the window) rather than a port of
  dplR's `rwi.stats.running`, which back-calculates `c.eff` from the number of
  valid within-tree correlations. A core whose within-tree correlations all fail
  `min_corr_overlap` in a window is no longer counted toward `c_eff` (dplR drops
  it too). Running EPS can differ from dplR by a few hundredths in sparsely
  replicated windows, enough to move the year a chronology crosses EPS 0.85.
- **`sfrcs`/`sfmrcs(method="difference")`** now return raw residuals like `rcs`
  (the signal-free iteration runs additively) unless `rescale_difference=True`,
  which is the previous always-on behaviour; `sfrcs`/`sfmrcs` accept the full
  `rcs` curve vocabulary (`fit`, `spline_period`, `nyrs0`, `pos_slope`, `min_n`,
  `preset`, ...) so `sfrcs(max_iterations=1)` equals `rcs` for every fit; their
  `return_info["rc"]` is the curve that actually produced `rwi`.
- **`edit_rwl` on a DataFrame** follows dplR's `rwl` method and never drops a
  measured ring (the frame gains or loses a year as needed; `fix_length` applies
  to Series input only and warns when it would drop a measurement); NaN or
  fractional years and negative widths raise clear errors.
- **Readers:** `readers(verbose=...)` and `read_crn(verbose=...)` silence the
  one-line success summary; all reader advisories (truncated values, short
  interior gaps, case-variant IDs, assumed encoding) are now `warnings`, not
  printed text; `skip_lines` counts physical lines; `header=True` forces the
  3-line ITRDB header when none is detected; `df.attrs["dplpy_blank_lines"]`,
  `dplpy_interior_gaps` and `dplpy_beyond_column` are capped summaries (totals
  kept) so a double-spaced file no longer slows every downstream call.
  `writers(prec=None)` (new default) writes at the precision the data were read
  at (`dplpy_precision`); `readers_url` emits a DeprecationWarning and forwards
  all keywords.
- `dpl.help("name")` (and `python -m dplpy help name`) prints a function's live
  signature and docstring; `dpl.help()` generates its overview from the installed
  package so it cannot go stale. `help` and `readme` are no longer in `__all__`
  (so `from dplpy import *` does not shadow the builtin `help`).
- `python -m dplpy --version` reports `dplpy.__version__` before distribution
  metadata; `plot()` closes its figure after `plt.show()` under a non-interactive
  backend.
- **Supported Python is now 3.11-3.15.** Added 3.14 and 3.15 to the tested
  matrix and dropped 3.10 (end-of-life). `requires-python` is now `>=3.11`, and
  the package advertises 3.11-3.15 via trove classifiers.
- **Lifted the `pandas<3` upper cap** (now `pandas>=2.2.2`). dplPy installs and
  runs on pandas 3.0, which is required on Python 3.15 (no pandas 2.x wheels are
  built for 3.15). The library code was already pandas-3.0 clean; two tests were
  updated for pandas 3.0's copy-on-write (a `to_numpy` array is now read-only) and
  its changed `type()` repr.

### Fixed
- **Interior gaps (unmeasured rings) are handled as gaps everywhere.** `rcs` and
  `mrcs` crashed on a series with an interior NaN, and `sfrcs`/`sfmrcs` shifted
  every year after the gap by the gap length; a gap now advances cambial age as
  the calendar does (CRUST/dplR behaviour), the gap years return NaN and later
  years are unchanged. Position-based detrending fits (`ModNegExp`, `NegExp`,
  `ModHugershoff`, `Hugershoff`, `GeneralExp`, the linear fits, `AgeDepSpline`)
  use the true ring position across a gap instead of compressing the age axis,
  so the old "gap compresses the age axis" warning is gone. Regional curves with
  zero-coverage cambial ages are fit on the true age axis. `xdate_floater` and
  `xdate_undated` keep the master on the dense year grid and correlate on
  pairwise-complete pairs (an all-NaN reference year no longer misdates a
  floater); `chron_stabilized` falls back to the sample depth where a window
  has no qualifying pair (dplR's rule) instead of returning NaN.
- **Unsorted or non-contiguous year indexes** were silently wrong in nine
  functions (`xdate`, `check`, `series_corr`, `interseries_corr`, `fill_internal`,
  `edit_rwl`, `xdate_floater`, `xdate_undated`, `normalize_for_crossdating`); the
  index is now validated (sorted with a warning, duplicates raise). `write_rwl`
  raises clearly on a non-contiguous index or duplicate column names.
- `chron()` keeps zero-depth years (std NaN, `samp_depth` 0, as dplR) so
  `chron_stabilized` no longer crashes on a chronology with a gap, and
  `chron(stabilize=..., ci=True)` no longer raises KeyError. `chron_ars` no
  longer emits statsmodels FutureWarnings or an IndexError on series shorter than
  the AR order (they become a NaN column with a warning), and
  `chron_ars(verbose=True)` prints the pooled ACF, coefficients, AIC and order it
  claimed to. `stabilize_chron(method="rbar")` raises when the window exceeds the
  chronology instead of silently degrading. `sens2` returns NaN for a zero-sum
  series instead of dividing by zero.
- All-NaN columns are dropped with a warning instead of crashing `detrend`,
  `powt`, `ssf`, `stats`, `chron_ars` and `ar_func`; short series fall back to
  the mean with a warning instead of raising from scipy or the banded solver;
  `ads` requires at least four points; `rcs(min_n=)` larger than any depth raises
  a clear error; `ssf(recode_zeros=True)` no longer modifies the caller's frame
  (a package-wide non-mutation test now covers every public transform); `bai_out`
  warns when `diam` would give negative basal area; `fill_internal(fill="Spline")`
  clips filled values at zero and warns which series were clipped.
- `xdate(ar_max=)` was ignored; `xdate_report` crashed on BC years and called the
  deprecated `make_plot=`; `check` and `xdate_report` suppressed every warning
  while running (now only the reader's provenance notes); an AR fit failure in
  `xdate_report` silently reported order 0 (now a warning and a blank); the
  COFECHA preset dropped series shorter than one window (now tested as a single
  segment with the matching critical value); `get_bins` warns when no bin fits
  so "0 problems" is never silent; `check`'s header-span check gave a false
  positive on BC headers; `check` reports the exact count of lines carrying
  characters beyond the last data column.
- `.crn` files with BC years round-trip: `write_crn` writes dplR's layout and
  `read_crn` detects it from the sign in column 6; `read_crn` warns when the
  first data decade disagrees with the header's year range. `write_crn`
  validates field widths (index below 10, depth 0 to 999), writes NaN depth as
  0, and packs decimal-degree coordinates and metric elevation from
  `SiteMetadata` back into the ITRDB DDMM/DDDMM and `%dM` fields. Tucson header
  metadata is routed by record number, latitude/longitude is found when a bare
  elevation precedes it (wa082), and a `#` inside a header line no longer drops
  it (nm580). CSV reading accepts a `year` column in any case or an unnamed index
  column, with clear errors otherwise. Whitespace-delimited and long-ID Tucson
  files are read in strict mode too via the whitespace fallback; series IDs over
  8 characters (7 with BC years) are truncated with a warning and kept unique;
  `write_rwl` warns about negative values (the format cannot carry them) and
  about values that collide with the stop code. The writer's gap sentinel
  (`-999`, `-99`, `-9`) is reported on read as a gap sentinel and no longer
  counted among dropped values. URL reads use the same encoding policy as local
  files and a 30 s timeout. TRiDaS disjoint series land on a contiguous year
  index and unrounded millimetres no longer claim `dplpy_precision=1.0`; a
  single-series LiPD raw-width table is recovered by `from_lipd`.
- `dpl.readme()` opened a dead URL; every docstring reference to the old
  `opendendro.org/dplpy-man` host (and the literal Sphinx `:func:` roles) now
  points at https://opendendro.github.io/dplPy/. `CITATION.cff` carries the
  v0.8.0 version DOI. `cli.help()` advertised a `writers(input=, output=)`
  signature that does not exist.

### Deprecated
- The old parameter names listed under *Changed* (`slide_period`, `seg_length`,
  `win_length`, `max_lag`, `lag_max`, `period`, `nyrs`, `spline_nyrs`,
  `stabilize_period`, `biweight_curve`, `biweight_crn`, `mad_threshold`,
  `difference`, `ssf(method=<curve>)`, `normalize_for_crossdating(method=)`)
  still work and emit a `DeprecationWarning`; they will be removed in v1.0.0.
- `readers_url` (use `readers(url)`).

### Removed
- `detrend.pick_first`, `curvefit.linear(bounds=)`, `chron.get_chron_info`,
  `chron.get_whitened_chron_info`, `chron_ars._get_first_min`, `rbar.get_running_rbar`,
  `rbar.mean_series_intercorrelation` and other internal duplicates folded into a
  single implementation (`autoreg._prewhiten`/`_levinson`/`_first_aic_min`,
  `rcs._align`/`_unalign`/`_resolve_po`/`_regional_curve`, `xdate._lag_table`/
  `get_bins`/`_bin_bounds`, `rbar.running_rbar_vector`/`pairwise_corr_matrix`,
  `stabilize._spline_component`, `smoothingspline.get_period`/`_caps`).
- The dead `src/dplpy/MANIFEST.in` (sdist contents unchanged).
- Python 3.10 support (reached end-of-life). Use Python 3.11 or newer.

### Performance
- `chron()` is vectorized (ca533: 84 ms to 7 ms); running rbar
  (`rbar.running_rbar_vector`) uses cumulative sums and `chron_ars(stabilize=)`
  computes it once per matrix (ca533: 3.8 s to 0.6 s; `chron_stabilized` 2.1 s to
  0.15 s); `xdate_floater(segmented=True)` is about 18x faster and
  `xdate_undated` about 1.7x; `common_interval(type="years")` keeps a running
  present-count (300 series: 28 s to under 1 s); `write_rwl` does one lookup per
  series instead of per cell; reader provenance attrs stay small so column access
  no longer deep-copies large lists.
- **Faster biweight master and horizontal detrend.** `tbrm_rows` (the biweight
  row-mean behind the crossdating master, `chron`, and `chron_ars`) now takes
  each row's median with a single sort instead of `np.nanmedian`, which on a
  NaN-filled matrix routes through numpy's masked-array sort path. The
  horizontal detrend used by `series_corr` and `interseries_corr` now reindexes
  each detrended series onto the full year index instead of an element-by-element
  `Series.combine`. Both are exact (same results to floating-point noise). On a
  356-series subfossil collection `interseries_corr` drops from about 38 s to
  13 s and `xdate` from about 43 s to 16 s.

### Added
- `chron(backcast=...)`; `chron_ars(prewhiten_method="burg")`;
  `rwi_stats`/`rwi_stats_running(ar_max, ar_method, first_aic_min)`;
  `xdate`/`series_corr`/`interseries_corr(ar_max, first_aic_min)`;
  `sfrcs`/`sfmrcs(rescale_difference=...)`; `detrend(plot="sequential")`;
  `rcs`-family `po` accepts a dict or Series and tolerates missing offsets (0
  with a warning); `curvefit` fitters accept a true age axis `t=`;
  `SiteMetadata.pack_crn_coordinates()`/`pack_crn_elevation()`;
  `df.attrs["dplpy_skipped_lines"]`; `[project.urls]` on PyPI.
- Test infrastructure: `tests/conftest.py` shared fixtures (`ca533_rwl`,
  `ca533_rwi`, `co021_rwl`, `DATA`), `tests/unit/test_cli.py`,
  `tests/unit/test_no_mutation.py` (a sweep asserting no public transform
  modifies its input), `tests/data/README.md` documenting the provenance of
  every test dataset, interior-NaN tests for every detrending and RCS function,
  and `pytest` now runs from a plain checkout (`pythonpath = ["src"]`).
- **TRiDaS (Tree Ring Data Standard) read and write.** `dpl.read_tridas`
  parses a TRiDaS XML file into a typed `TridasProject` (ring-width tables,
  derived chronologies, undated/floating series, and the full
  project/object/element/sample/radius/series metadata); a standalone
  `dpl.write_tridas` emits a TRiDaS 1.2.2 file from a ring-width frame and/or
  a chronology plus a `TridasMetadata`, with mandatory-field validation.
  `dpl.readers()` now detects a `<tridas>` file and returns the ring-width
  frame for the common single-table case (a multi-table or chronology file
  points you to `read_tridas`), and `dpl.writers(format="tridas")` delegates
  to the writer. Modeled on dplR's `read.tridas`/`write.tridas` but written as
  a stdlib ElementTree walk rather than a port. Missing values are read as
  NaN (never 0, so a measured 0 stays a real locally-absent ring); units
  convert to millimetres and reuse dplPy's precision model; chronologies map
  to `derivedSeries` (sample depth round-trips as each value's `count`).
  v1 keeps duplicate series separate with a warning rather than merging.
- **`dpl.edit_rwl`** inserts, deletes, or changes a single ring in a series,
  selected by `action="insert"/"delete"/"change"` -- the correction that pairs
  with the crossdating checks (find a misdated ring with `xdate`/`series_corr`/
  `check`, fix it here). It folds dplR's `insert.ring` and `delete.ring` into one
  dispatch function and adds a value edit, keeps dplR's `fix_last`/`fix_length`
  semantics, and always returns a copy (it never edits in place).
- **`dpl.chron` can return bootstrap confidence intervals** for the standard
  chronology with `ci=True`, adding `lower_ci` and `upper_ci` columns (a
  normal-approximation interval from resampling each year's series; `conf`, `R`,
  and `seed` tune it). A confidence interval is part of building a chronology, so
  it is a builder option rather than a separate function. Years with fewer than
  two series get NaN limits with a warning. The interval covers the standard
  mean-value chronology only; residual and ARSTAN chronology intervals are left
  to a future full-pipeline bootstrap.
- **`dpl.check`** runs integrity checks over a ring-width collection and
  returns the findings as a table, one row per finding, each with a stable
  check id and a severity (`error`/`warning`/`note`). It is the data
  counterpart to `report`: `report` prints what a file looks like, `check`
  reports what is wrong with it, so a report over one file and a sweep over a
  whole archive are the same call. A sweep runs over a directory or a list of
  paths, and `summary=True` returns one row per file for ranking an archive.
  Modeled on dplR 1.8.0's `rwl.check` (A. G. Bunn) and built on dplPy's own
  reader, spline, and crossdating pieces, so it keeps dplPy's read tolerance.
  Checks cover collection structure, per-series shape (internal gaps,
  duplicates, zero variance, repeated-value runs that expose a constant used as
  a not-measured placeholder), implausible values and precision, locally-absent
  rings, crossdating against a leave-one-out master (lag, poor-fit outliers,
  weak common signal), raw-file hygiene (mixed line endings, tabs, non-ASCII,
  missing final newline, header span), and read provenance from the reader's
  `df.attrs`. Every check runs guarded, so a check that fails becomes a finding
  rather than stopping the run.
- **`dpl.mrcs`** performs multiple-curve Regional Curve Standardization (a port
  of CRUST's multiple-RCS option, Melvin & Briffa 2014, Part 2): trees are sorted
  by relative growth rate (or age/diameter) and split into `n_curves` equal-count
  classes, each standardized by its own regional curve -- correcting the slope
  ("modern sample bias") a single average curve injects when the sample's
  growth-rate mix changes over calendar time. `reset_means=True` (default)
  rescales each tree back to its one-curve mean to recover the low-frequency
  variance that multiple RCS otherwise removes (CRUST "mean single"). Each class
  curve is built by `rcs`, so `mrcs` shares its full `fit`/`method`/`period`
  vocabulary and `mrcs(n_curves=1)` is identical to `rcs`.
- **`dpl.sfmrcs`** adds signal-free multiple-curve RCS: the signal-free iteration
  (as in `sfrcs`) wrapped around the growth-rate curve allocation (as in `mrcs`),
  a port of CRUST's signal-free multiple-RCS. `reset_means=True` (default)
  restores each tree's one-curve mean; `sfmrcs(n_curves=1)` is identical to
  `sfrcs`.

### Documentation
- The docs index states that the site tracks the development branch and names
  the released version. `xdate_undated` is documented (API reference,
  floating-series guide, README, dplR map). The internal-gaps and detrending
  guides describe the true-age-axis behaviour; the chronology, variance
  stabilization and fidelity pages describe the ARSTAN chronology construction
  and the AR order-selection difference from R's `ar()`. Guides, README and
  notebooks use the new parameter names and return shapes. Rewritten
  `dev-instructions.md` (src layout, setup, tests, lint, docs build, optional
  `mike` versioned docs) and `release-instructions.md` (CHANGELOG step, version
  DOI step, PyPI trusted publishing); new `tests/integs/README.md`.
- Packaging and CI: SPDX `license = "GPL-3.0-or-later"` with `license-files`
  and `setuptools>=77` (the wheel metadata no longer embeds the GPL text); the
  release workflow uses trusted publishing, `actions/checkout@v4` and
  `softprops/action-gh-release@v2` with the tag and notes derived from the
  release branch and CHANGELOG; CI installs the `[lipd]` extra on Python 3.13
  and runs `ruff check`; the pandas `<3` cap is removed from the 3.11/3.12
  requirement files to match `pyproject.toml`; `environment.yml` installs the
  package in editable mode with `pytest-cov` and `ruff`.

## [0.8.0] - 2026-10-12

### Changed
- **BREAKING -- `dpl.rcs` and `dpl.sfrcs` inputs are harmonized with
  `dpl.detrend` (dplPy 0.8.0).** The regional-curve inputs now use the same
  vocabulary as per-series detrending:
  - `method=` now selects the **arithmetic** (`"ratio"` / `"difference"`),
    matching `detrend()`. The boolean `ratios=` is removed (`ratios=True` ->
    `method="ratio"`, `ratios=False` -> `method="difference"`).
  - the regional **curve** is chosen with `fit=` (old `method="caps"` ->
    `fit="Spline"`, `method="ads"` -> `fit="AgeDepSpline"`). `fit=` accepts the
    full `detrend()` curve vocabulary -- `Spline`, `AgeDepSpline`, `ModNegExp`,
    `NegExp`, `ModHugershoff`, `Hugershoff`, `GeneralExp`, `LinearAny`,
    `LinearNegative`, `Mean` -- plus `None` (no smoothing).
  - the default regional curve is now **`fit="AgeDepSpline"`** (ARSTAN's
    recommended RCS curve), previously the fixed spline. Results of calls that
    did not name a curve will change; use `fit="Spline", period=0.1` to
    reproduce the previous default exactly.
  - spline stiffness is set with `period=` (default `0.1`, via the same
    `get_period` rule as `detrend`), replacing `nyrs=`. `period=-10` gives the
    ARSTAN percent-of-length spline.
  - `pos_slope=` now defaults to **`False`** (a non-increasing old-age end, as in
    ARSTAN's "non-increasing end fitting" and CRUST's "no rise in the final
    third"), previously `True`.
  - the figure switch is now `plot=` (see Deprecated).
- **BREAKING -- the pith offset is now a true offset.** `pith_offset` (in the
  `po=` table and in `po_to_wc` / `wc_to_po`) is now the number of rings between
  the pith and the first measured ring, so **0 means the pith is present**
  (previously 1, the cambial age of the first ring, following dplR). This makes
  dplPy's pith offset identical to ARSTAN's `nshft`, so an ARSTAN pith-offset
  file is usable directly; a dplR `po` table must have 1 subtracted first.
  `po_to_wc` now returns `n_missing_heartwood = pith_offset` (was `- 1`) and
  `wc_to_po` returns `missing + unmeasured` (was `+ 1`).
- **The plotting keyword is standardized to `plot=` across the package**
  (`rcs`, `sfrcs`, `xdate`, `series_corr`, `xdate_floater`, `xdate_undated`,
  `common_interval`), matching `detrend` / `chron` / `powt`.

### Deprecated
- **`make_plot=` is deprecated in favor of `plot=`** everywhere it was used. It
  still works for one release and emits a `DeprecationWarning`; it will be
  removed in a future version.

### Added
- **`dpl.rcs`** now warns when the regional curve is non-positive at one or
  more cambial ages (where ratio indices would blow up or go negative). ARSTAN
  halts in this case; dplPy warns and continues so batch runs are uninterrupted,
  and points to a stiffer smoothing (a larger `period`), `fit='AgeDepSpline'`,
  or `method='difference'`.
- **`dpl.rcs`** gains an opt-in `rescale_difference=` argument (default `False`).
  When `True` and `method='difference'`, each series' difference (residual) index is
  rescaled onto the mean and standard deviation of that series' own ratio index,
  so difference and ratio members share the ~1 scale -- the per-series version of
  ARSTAN's `calind` convention. Default `False` keeps the current residuals
  centred on 0, so existing calls are unchanged.
- **`dpl.rcs` can now fit the regional curve with any `detrend()` curve or with
  no smoothing.** `fit=None` uses the age-aligned mean itself as the regional
  curve (ARSTAN's "no smooth" / CRUST's unsmoothed option), and the parametric
  growth curves from `curvefit` (`ModNegExp`, `NegExp`, `Hugershoff`, ...) can be
  fitted to the regional mean, mirroring how ARSTAN and CRUST build the RC.

- **`dpl.readers`** gains an `encoding=` argument and no longer crashes on a file
  that is not valid UTF-8. A non-UTF-8 byte (e.g. an accented investigator name in a
  header comment, common in European ITRDB files) previously raised
  `UnicodeDecodeError`; the reader now falls back to latin1 (lossless) with a warning,
  records the encoding in `df.attrs["dplpy_encoding"]`, and accepts an explicit
  `encoding=` to set it exactly. Prompted by the parallel fix in dplR 1.8.0.
- **`dpl.readers`** records each series' measurement precision (0.01 or 0.001 mm) in
  `df.attrs["dplpy_precision"]`, with `df.attrs["dplpy_mixed_precision"]` flagging a
  file that mixes the two.

- **`dpl.xdate_undated`** is a new function that crossdates a set of *undated*
  ("floating") ring-width series against one another and assembles them into a single
  relatively-dated floating chronology (earliest ring anchored at `anchor_year`,
  default 1). It is the dplPy analogue of COFECHA's UFLOAT — the routine COFECHA runs
  when only an undated-series file is supplied — for material that overlaps in time but
  does not crossdate into any absolutely-dated chronology (e.g. sub-fossil stands).
  It does the all-pairs matching (`result["pairwise"]`), then grows a master chronology
  (seed on the strongest pair, add each series at its best position against the running
  mean), and runs a leave-one-out verification with an isolation score so weak/ambiguous
  placements are flagged for review (`result["verify"]`) rather than silently misplaced.
  Returns the pairwise table, the per-series relative offsets, the placed ring-width matrix,
  the mean floating chronology, the join order, the verification table, and any unplaced
  series; with `make_plot` it draws a placement chart (each series on the relative axis,
  colored by the isolation factor) over the mean chronology.
- **`dpl.xdate_floater`** gains a segment-consensus dating mode (`segmented=True`),
  in the spirit of COFECHA's UDATE: each overlapping segment of the floater is dated
  against the master independently and the implied youngest-ring years are compared.
  A single dominant year means the floater is internally consistent; a coherent
  one-year step between older and younger segments flags an internal dating error a
  whole-series slide cannot reveal (older rings dating one year early is the
  signature of a missing/locally-absent ring, one year late a false ring). The
  result gains `result["segments"]` (per-segment implied end year and correlation),
  `result["consensus"]` (a vote count of each segment's single-best implied end year,
  matching the staircase plot), `result["segment_candidates"]` (each segment's top few
  placements, controlled by `segment_topk=`, so a near-tie between dates is visible
  rather than hidden behind the winner), and `result["internal_error"]` (a dict
  describing a detected missing/false ring, or `None`). At least four segments are needed to test
  for an internal error, so short floaters report the per-segment dates without a
  verdict, and the printed summary flags a weak whole-series match rather than
  asserting a confident date. A detected error prints a banner regardless of
  `verbose`, and `make_plot` adds a staircase diagnostic (implied end year vs. ring
  number, counted from the pith).
- **`dpl.detrend`** now emits a warning when a series has an *interior* gap
  (`NaN` between its first and last measured ring) and the chosen fit is fitted
  against reconstructed ring position — the parametric growth curves
  (`ModNegExp`, `NegExp`, `ModHugershoff`, `Hugershoff`, `GeneralExp`), the
  linear fits, and `AgeDepSpline`. For those, an interior gap compresses the
  cambial-age axis after the gap; `fit="Spline"` (real-year `csaps`) and
  `fit="Mean"` are unaffected and do not warn. The warning names the series and
  points to `dpl.fill_internal` and the new *Handling internal gaps* guide. dplPy
  never fills gaps silently — the choice is left to the user.

### Changed

- **`dpl.xdate_floater`** and **`dpl.xdate_undated`** now prewhiten with the
  COFECHA/ARSTAN AR-order convention by default (`ar_max=10`, `first_aic_min=True`:
  cap the order at 10 and take the first local AIC minimum) instead of dplR's
  global-AIC rule with a `floor(10*log10(n))` ceiling. The dplR rule can select very
  high orders (e.g. 23 for a ~450-ring series) and trim that many early rings before
  matching, shrinking the usable overlap; the COFECHA convention keeps far more early
  rings (that series drops to order 6). Both parameters are exposed, so
  `ar_max=None, first_aic_min=False` restores the previous dplR-default behavior. This
  changes `xdate_floater`'s reported t-values slightly; recovered dates are unaffected.
- **`dpl.detrend`** now falls back from a non-positive spline fit to detrending by the
  series mean *only* for `method="ratio"` (division, which cannot use a non-positive
  curve). For `method="difference"` (subtraction) the fit is kept, since a non-positive
  curve subtracts fine — the correct behavior for log-transformed or isotope series
  (cf. dplR issue #22). The regular `fit="Spline"` now uses this same guard, so a
  non-positive spline in ratio mode falls back to the mean (previously it produced
  negative indices silently) instead of passing them through.

### Fixed

- **`dpl.xdate_floater`** no longer crashes building `result["combined"]` when the
  floating series shares its name with a column already in the reference collection
  (e.g. a leave-one-out test that did not drop the column); the placed copy is
  suffixed on the overlap instead.
- **`dpl.detrend`** curve-fit fallbacks are now uniformly announced: every fall back to
  a linear fit or the series mean emits a warning, including a previously-silent
  degenerate case in the `GeneralExp` fit.
- **`dpl.interseries_corr`** no longer crashes (or returns a spurious ±1) on a series
  that overlaps the master by fewer than 3 years, or has no values at all; such a
  series now gets an `NaN` correlation and is named in a warning, matching
  `dpl.series_corr` and dplR 1.8.0. The collection mean skips the `NaN`s.

### Documentation

- New User Guide vignette **Handling internal gaps** (`docs/guide/internal-gaps.md`):
  what an interior `NaN` is versus a measured zero, why `Spline`/`Mean` tolerate a
  gap while the position-based fits are age-compressed by it, how ARSTAN infilled
  gaps automatically (and why dplPy deliberately does not), and how to detect,
  decide, and act with `fill_internal` (including `fill="ARSTAN"`). The
  `detrend` docstring and the detrending guide note were corrected: `AgeDepSpline`
  is *not* gap-safe (it fits on ring position, like the parametric curves); only
  `Spline` spans a gap in calendar time.
- Documented the **pith-offset convention** for `dpl.rcs`/`dpl.sfrcs` (and in the
  detrending guide): `pith_offset` is the cambial age of the first measured ring
  (1 = pith present) = ARSTAN's `nshft` + 1, so an ARSTAN pith-offset file must be
  shifted by 1. Added a guide section comparing dplPy's single-curve RCS to ARSTAN
  (regional-curve smoothing options, difference-index scaling, non-positive-curve
  behavior, pith convention).



## [0.7.0] - 2026-09-26

### Added

- **`dpl.readers`** now recognizes a core whose series ID changes letter case
  partway through a file (e.g. `FDD08A` → `FDD08a`). When the case-variant blocks
  have disjoint years and one lacks a stop marker — a strong sign of one core split
  by a case typo — they are merged into a single series under `join=True` (recorded
  in `df.attrs["dplpy_case_merged"]`), which also resolves the spurious
  "no stop marker" precision warning. Case-variant IDs that are not merged (under
  `join=False`, or when both blocks are properly terminated) are flagged in
  `df.attrs["dplpy_case_variant_ids"]`.
- **`dpl.readers`** advisory for characters beyond the last data column (col 72),
  most importantly a value that overflowed its field and was truncated, recorded in
  `df.attrs["dplpy_beyond_column"]`.
- **`dpl.xdate`** prints a COFECHA-style summary header (number of series, master
  span, total rings, intercorrelation, mean sensitivity, problem segments, mean
  series length) at the top of its output, and returns it as `res["summary"]`. The
  default path now also returns `n_problems`.
- **`dpl.xdate_floater`** accepts a single chronology as a pandas `Series` or a
  one-column `DataFrame` (e.g. a column from `dpl.read_crn`) as the reference, in
  addition to a full ring-width collection.
- **`dpl.xdate_floater`** auto-derives `series_name` from a named `Series` or a
  one-column `DataFrame` when it is not passed explicitly, so outputs and the plot
  title show the real series name.
- **`dpl.xdate_floater`** raises an impossible/future-date alarm when the best-fit
  date falls after the present calendar year: `best["date_warning"]`, a printed
  banner (even with `verbose=False`), and a red banner on the returned plot.
- **`dpl.xdate_floater`** plot adds a fourth panel overlaying the reference
  chronology against the best-placed floating series over the dated overlap.
- **`dpl.xdate_report`** accepts an already-loaded ring-width `DataFrame` (with an
  optional `name=` label) in addition to `.rwl` file paths and lists of them.
- **`dpl.detrend`** exposes ARSTAN curve-fit options: `NegExp`, `Hugershoff`,
  `GeneralExp`, `LinearAny`, and `LinearNegative` (alongside the existing dplR-style
  fits).
- **`dpl.powt`** reports ARSTAN-like spread-vs-level and skewness statistics with an
  accompanying plot.
- **`dpl.chron_ars`** supports ARSTAN-style backcasting.
- Example notebook: `notebooks/readers_demonstration.ipynb`.

### Changed

- **`dpl.report`** lists absent rings (zero values) and internal NA (missing
  measurements) with per-series and total counts, clearer section labels, and
  `(none)` for an empty section.
- **`dpl.interseries_corr`** returns `(mean_corr, per_series_df)` — the
  collection-wide mean first, then the per-series table.
- **`dpl.xdate` / `dpl.xdate_report`** crossdating outputs refined from notebook
  testing: mutually-exclusive A/B segment flags (COFECHA-style), overall `rho`
  reported to three decimals, and `xdate_report`'s destination controlled by
  `output=` (`"screen"`/`"file"`/`"both"`).
- **`dpl.xdate_floater`** `series_name` now defaults to `None` (derive from the
  series, else `"Unknown"`), and the returned plot's layout was reorganized (the
  T-value distribution moves beside the sliding-T panel to make room for the new
  overlay).
- **`dpl.sfrcs`** (Signal-Free RCS) now uses a simple (not biweight) mean by
  default, following Melvin.
- **`dpl.readers`** no longer warns about harmless blank input lines; their
  positions are recorded in `df.attrs["dplpy_blank_lines"]` instead.

### Fixed

- **`dpl.xdate_floater`** no longer raises an `IndexError` when the floating series
  is longer than the reference master (the master lying entirely inside the
  floater); the sliding-offset search was rewritten to handle every overlap regime
  and reproduces prior results offset-for-offset when the floater is the shorter
  series.
- **`dpl.xdate_floater`** plot no longer balloons with white space when the best
  match is poor: the p-value panel's inverted-log axis always extends past the
  `p = 0.05` / `p = 0.0001` thresholds, so their reference lines and labels stay
  inside the panel instead of landing far above it.
- **`dpl.xdate_report`** given an in-memory `DataFrame` no longer misfires by
  treating the frame's columns as file paths (the previous
  "No such file or directory" flood); it is now handled as one collection.
- **`dpl.xdate(preset="COFECHA")`** no longer emits spurious early/anchored segments
  for a series that extends before the multi-series portion of a collection, by
  clipping each series to the span where two or more series overlap (matching
  COFECHA).
- **`dpl.readers`** with `join=False` reports the split of disjoint same-ID blocks
  correctly.

### Documentation

- Added narrative guides and reference pages (mkdocs + mkdocstrings), and updated
  the README, citation/DOI metadata, and background references.

[Unreleased]: https://github.com/OpenDendro/dplPy/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/OpenDendro/dplPy/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/OpenDendro/dplPy/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/OpenDendro/dplPy/compare/v0.6.0...v0.7.0
