# Reading & writing data

dplPy reads the standard tree-ring file formats into a pandas `DataFrame` indexed
by year, with one column per measurement series — the shape every other function
expects. This guide covers loading ring widths, loading chronologies, combining
collections, and writing data back out.

## Ring-width files with `readers`

`readers` loads Tucson (`.rwl`), CSV, and compact-format ring-width files:

```python
import dplpy as dpl

rwl = dpl.readers("ca533.rwl", header=True)
```

```text
ca533.rwl successfully extracted as rwl file with 34 series covering the period from 626 to 1983
```

Key arguments:

- `header` — set `True` when a Tucson file carries the three ITRDB header lines,
  so they are read as metadata rather than data.
- `strict` (default `True`) — mirror dplR's strict validation: dplPy warns and
  then raises on malformed rows, naming the offending series and year. Set
  `strict=False` to fall back to a best-effort parse that returns `None` on
  unrecoverable files instead of raising.
- `join` (default `True`) — when a series ID appears in more than one block,
  join the pieces into a single column. dplPy reports duplicated and
  non-overlapping same-name series so you can see what was merged.
- `format` — normally auto-detected; override with `"csv"`, `"rwl"`, etc. if
  needed.

To load directly from a URL (for example an ITRDB file), use `readers_url`:

```python
rwl = dpl.readers_url("https://www.ncei.noaa.gov/pub/data/paleo/treering/.../ca533.rwl")
```

!!! note "NOAA template files"
    dplPy recognizes the `*-noaa.rwl` NOAA-template files and refuses them with
    an informative error — they are not decadal Tucson data. Use the plain
    `.rwl` file (the same name without `-noaa`).

## Chronology files with `read_crn`

`read_crn` reads Tucson chronology (`.crn`) files — the standardized site
chronologies produced by ARSTAN and friends:

```python
crn = dpl.read_crn("ca533.crn")
```

It handles the standard single-chronology ITRDB file, the "combined" multi-block
files (an ARSTAN run's stacked std/res/ars chronologies for one site), and files
that concatenate many sites' chronologies. Bunched negative (BC/BCE) years parse
correctly, so long chronologies read as expected. Use `split_by_site=True` to
return separate frames per site block.

## Combining collections with `combine_rwl`

`combine_rwl` merges ring-width datasets onto a common year index, following
dplR's `combine.rwl` (duplicate series IDs are kept as-is, not renamed):

```python
both = dpl.combine_rwl(site_a, site_b)
```

## Writing data with `writers`

`writers` writes a ring-width or chronology `DataFrame` back to disk. Specify the
output `format` — `"rwl"` (Tucson), `"crn"`, or `"csv"`:

```python
dpl.writers(rwl, label="ca533_clean", format="rwl", prec=0.001)
```

For `.rwl` output, `prec` sets the measurement precision (`0.001` writes values
×1000 with a `-9999` end marker; `0.01` writes ×100 with a `999` marker), and
`gaps` controls how a true interior gap (a NaN inside a series' span) is encoded.
See the [`writers` reference](../reference/io.md#writers) for the full option set.

## Quick metadata

`metadata` extracts the ITRDB header fields (site, species, investigators,
location) from a headered Tucson file without loading the measurements, and
`summary`/`report` give per-series overviews. See the
[Input & output reference](../reference/io.md).
