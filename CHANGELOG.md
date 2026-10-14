# Changelog

All notable changes to dplPy are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and dplPy aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This changelog was started at the `v0.6.0` tag as the project moves toward a 1.0
release; the history of earlier releases (`v0.1.5`–`v0.6.0`) lives in the git tags
and commit log.

## [Unreleased]

### Changed
- **Supported Python is now 3.11-3.15.** Added 3.14 and 3.15 to the tested
  matrix and dropped 3.10 (end-of-life). `requires-python` is now `>=3.11`, and
  the package advertises 3.11-3.15 via trove classifiers.
- **Lifted the `pandas<3` upper cap** (now `pandas>=2.2.2`). dplPy installs and
  runs on pandas 3.0, which is required on Python 3.15 (no pandas 2.x wheels are
  built for 3.15). The library code was already pandas-3.0 clean; two tests were
  updated for pandas 3.0's copy-on-write (a `to_numpy` array is now read-only) and
  its changed `type()` repr.

### Removed
- Python 3.10 support (reached end-of-life). Use Python 3.11 or newer.

### Performance
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

[Unreleased]: https://github.com/OpenDendro/dplPy/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/OpenDendro/dplPy/compare/v0.6.0...v0.7.0
