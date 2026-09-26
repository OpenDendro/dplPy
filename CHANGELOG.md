# Changelog

All notable changes to dplPy are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and dplPy aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This changelog was started at the `v0.6.0` tag as the project moves toward a 1.0
release; the history of earlier releases (`v0.1.5`–`v0.6.0`) lives in the git tags
and commit log.

## [Unreleased]

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

[Unreleased]: https://github.com/OpenDendro/dplPy/compare/v0.6.0...HEAD
