# CooRecorder .pos files

CooRecorder (Cybis Elektronik & Data AB) stores a measured series as a `.pos`
file: the x, y coordinates of each ring boundary along the measurement path. The
usual way into dplPy would be to export those files to Tucson `.rwl` from CDendro
and read the `.rwl`, but that export rounds the widths and drops the extra
information the `.pos` file holds. `dpl.read_pos` reads the `.pos` file directly,
at full precision, and keeps what the export throws away: earlywood and latewood
widths, the pith coordinate and distance to pith, point labels and comments, and
the raw coordinates for quality control.

`read_pos` requires CooRecorder 7.8 or later, the first version to record the
pith coordinate.

## Reading one file or a whole directory

```python
import dplpy as dpl

proj = dpl.read_pos("cores/AGL3_01A.pos")   # one file
proj = dpl.read_pos("cores/")               # or every .pos under a directory
```

`read_pos` returns a `PosCollection`. The whole-ring widths are a familiar
year-by-series `DataFrame`, ready for the rest of dplPy:

```python
rwl = proj.rwl()          # whole-ring widths (mm), years x series
ew  = proj.earlywood()    # earlywood widths, where seasonwood points were marked
lw  = proj.latewood()     # latewood widths
```

The per-series information CooRecorder recorded is in `proj.attributes` (one row
per series: distance to pith, radius, outer and inner year, comment, CooRecorder
version), the point-by-point geometry is in `proj.coordinates`, and any files
that could not be read are listed in `proj.not_read` with a reason, so one bad
file never aborts a directory read.

If you only want the ring widths, `readers` takes you straight there — a `.pos`
file (or a directory of them) is detected automatically, and the attributes ride
along on `df.attrs["dplpy_pos_attributes"]`:

```python
rwl = dpl.readers("cores/")                 # whole-ring DataFrame
rwl = dpl.readers("cores/AGL3_01A.pos", format="pos")
```

## From CooRecorder to a pith offset

A `.pos` file often records an estimated distance to pith even when the pith ring
itself was never measured. `dpl.yrs_to_pith` turns that distance into a pith
offset — the estimated number of rings between the pith and the innermost
measured ring — which `rcs` and `po_to_wc` consume directly:

```python
proj = dpl.read_pos("cores/")
po   = dpl.yrs_to_pith(proj.rwl(), proj.attributes[["series", "distance_to_pith"]])
rwi  = dpl.rcs(proj.rwl(), po[["series", "pith_offset"]])
```

`yrs_to_pith` divides the distance to pith by a mean innermost ring width. With
`method="dist"` (the default) it averages as many innermost rings as fit within
that distance; with `method="rings"` it averages a fixed `n_rings`.

## Things to know

- **Dates.** The outer date you assigned in CooRecorder is given to the outermost
  ring and years count inward. A file with no outer date is read with *relative*
  years, which is exactly what you want for an undated floating series headed for
  [`xdate_floater`](floating-series.md). One caveat you cannot automate away:
  CooRecorder sometimes stores an outer date you did not intend, and nothing in
  the file distinguishes that from a deliberate one — so sanity-check the assigned
  years, for example by plotting the series, before trusting them.
- **Cracks and gaps.** dplPy measures around a crack (gap) marker rather than
  through it, so a crack does not count as growth. This can make a width differ
  slightly from CDendro's own `.rwl` export for a series with marked gaps.
- **Earlywood and latewood.** Where you marked seasonwood boundaries, the ring is
  split into latewood (the outer part) and earlywood (the inner part), and
  earlywood + latewood equals the whole ring. In the ITRDB `.rwl` convention these
  are written as separate files with an `e` or `l` after the series number; here
  they are simply three frames you can use or ignore.
- **Point order.** If points were saved out of order (a known CooRecorder export
  glitch), the widths for that series will be wrong. Reliable automatic detection
  is hard, so `read_pos` does not guess; instead it returns `proj.coordinates`,
  which you can plot to check that the points march cleanly along the core.

See the [Input & output reference](../reference/io.md) for the full `read_pos`
and `PosCollection` signatures, and [Utilities](../reference/utilities.md) for
`yrs_to_pith`.
