# LiPD interchange

[LiPD](https://lipd.net) (Linked Paleo Data) is a standard container format for
paleoclimate datasets. dplPy can export a chronology (with its underlying ring
widths and metadata) to a `.lpd` file and read one back, so dplPy results travel
into the wider paleo ecosystem.

## Installing the extra

LiPD support is built on [`pylipd`](https://pypi.org/project/pylipd/) and is an
optional extra, since most users don't need it:

```bash
pip install "dplpy[lipd]"
```

## Exporting with `to_lipd`

Build a chronology as usual, then write it to a `.lpd` file. Passing the original
`rwl` stores the ring-width measurements alongside the chronology:

```python
import dplpy as dpl

rwl = dpl.readers("ca533.rwl", header=True)
rwi = dpl.detrend(rwl, fit="Spline")
crn = dpl.chron(rwi)

dpl.to_lipd(crn, "ca533", rwl=rwl, dsname="CA533", archive_type="Wood")
# -> writes ca533.lpd
```

Useful arguments include `dsname` (the LiPD dataset name), `archive_type`,
`column` (which chronology column to store), and `metadata` / `publication` /
`provenance` for richer records. Site metadata (species, location,
investigators) can be supplied via a [`SiteMetadata`](../reference/io.md#sitemetadata)
object or a metadata dict.

## Reading with `from_lipd` and `lipd_metadata`

`from_lipd` reads a `.lpd` file back into dplPy structures:

```python
data = dpl.from_lipd("ca533.lpd")
data.keys()
# dict_keys(['dsname', 'rwl', 'chronology', 'chronologies', 'metadata'])
```

The returned dictionary contains:

- `rwl` — the ring-width `DataFrame` (if it was stored),
- `chronology` — the primary chronology (columns `trsgi`, `sampleCount`),
- `chronologies` — a dict of all stored chronology types (e.g. `standard`),
- `metadata` — the site/dataset metadata,
- `dsname` — the dataset name.

To read just the metadata without loading the measurements, use
`lipd_metadata("ca533.lpd")`.

## Round trip

A chronology exported with `to_lipd` and re-read with `from_lipd` returns the same
chronology values and ring widths, so `.lpd` is a lossless carrier for a dplPy
analysis:

```python
crn = dpl.chron(dpl.detrend(rwl, fit="Spline"))
dpl.to_lipd(crn, "ca533", rwl=rwl, dsname="CA533", archive_type="Wood")
back = dpl.from_lipd("ca533.lpd")
back["chronologies"]["standard"].tail()   # same chronology, round-tripped
```

See the [utilities reference](../reference/utilities.md#to_lipd) for every
parameter.
