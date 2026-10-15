# Test data

Datasets used by the unit and integration tests (`tests/unit`, `tests/integs`)
and by the example notebooks. Tests reach them through the `DATA` fixture in
`tests/conftest.py` (`tests/data`), and the session-scoped `ca533_rwl`,
`ca533_rwi` and `co021_rwl` fixtures load the two workhorse collections once.

Most files are measurement series downloaded from the International Tree-Ring
Data Bank (ITRDB, hosted by NOAA NCEI at
<https://www.ncei.noaa.gov/products/paleoclimatology/tree-ring>), named by their
ITRDB site code. They are redistributed here unchanged unless noted; the ITRDB
is the authoritative source. The rest are small synthetic files, reference
outputs from R/dplR, or deliberately malformed files that exercise the reader's
error handling.

## `rwl/` — Tucson-format ring widths

| File | Provenance | Notes |
|---|---|---|
| `ca533.rwl` | ITRDB **ca533**, Campito Mountain, California, USA; bristlecone pine (*Pinus longaeva*, PILO); 34 series, 626–1983. | The workhorse dataset (also in dplR as `ca533`). No header lines. |
| `co021.rwl` | ITRDB **co021**, Mesa Verde, Colorado, USA; Douglas-fir (*Pseudotsuga menziesii*, PSME); 35 series, 1176–1963. | Also in dplR as `co021`. No header lines. |
| `wa082.rwl` | ITRDB **wa082**, Hurricane Ridge, Washington, USA; Pacific silver fir (*Abies amabilis*, ABAM); F. Schweingruber; 23 series, 1698–1983. | Also in dplR as `wa082`. Three header lines. Used in the `xdate` bin-floor regression. |
| `ca667.rwl` | ITRDB **ca667**, Sheep Mountain update, California, USA; bristlecone pine (PILO); M. Salzer & M. Hughes; −2649–2005. | Large, long, headered file (BC years) for the readers/plot integration tests. |
| `cana209.rwl` | ITRDB **cana209**, Canada; series `EGL1xx`; 1820–2000s. | Clean header-less file for the no-header read/write round-trip. |
| `cana295.rwl` | ITRDB **cana295**, Bison, Canada; white spruce (*Picea glauca*, PCGL); A. Bunn, A. Lloyd, L. Berner; 1933–2008. | Short, headered. |
| `cana326_error.rwl` | Derived from ITRDB **cana326**, Yukon composite, Canada; white spruce (PCGL); R. D'Arrigo, R. Wilson, G. Jacoby; 1067–2002. | **Deliberately broken.** Several series carry IDs with embedded spaces and an apostrophe (`SF 31 E'`) that run into the year field, so the record cannot be tokenised on whitespace. Used to exercise the strict (`ValueError`) and `strict=False` salvage paths. Do not "fix" it. |
| `kyrg014.rwl` | ITRDB **kyrg014**, Kok-Kiya, Kyrgyzstan; Schrenk's spruce (*Picea schrenkiana*, PCSH); O. Solomina; 1551–2005. | Contains a series (`kok3a`) whose precision changes mid-series (a `-9999` segment followed by a `999` segment); the reader must refuse to read it at one precision. |
| `nc004.rwl` | ITRDB **nc004**, North Carolina, USA; series `1100xx`; 1780–1980s. | Header-less. |
| `nm046.rwl` | ITRDB **nm046**, New Mexico, USA; series `6440xx`; 1681–1970s. | Small header-less file. |
| `nm580.rwl` | ITRDB **nm580**, El Malpais update, New Mexico, USA; Douglas-fir (PSME); H. Grissino-Mayer, D. Stahle, M. Therrell; −136–2004. | Total ring width. The header contains `UAFACC#=04-223`; the `#` once made the reader drop the first data row (regression test). |
| `nm580e.rwl` | ITRDB **nm580e**, as above. | Earlywood width (`E_` flag in the header). |
| `nm580l.rwl` | ITRDB **nm580l**, as above. | Latewood width (`L_` flag), edited. |
| `russ301.rwl` | ITRDB **russ301**, Bograd, Russia; Scots pine (*Pinus sylvestris*, PISY); E. Babushkina et al.; 1847–2018. | Headered; non-standard header spacing. |
| `th001.rwl` | ITRDB **th001**, Mae Hong Son, Thailand; teak (*Tectona grandis*, TEGR); B. Buckley, B. Cook, W. Wright; 1558–2005. | Three-line header (auto-detection test); series `PATUNG` carries a spurious `-2599` value that must become NaN with a warning. |
| `tx042.rwl` | ITRDB **tx042**, Big Bend National Park (incl. Camp Springs), Texas, USA; Douglas-fir (PSME); E. Cook, N. Montagu; 1473–1992. | Used by the `metadata()` extraction tests. |
| `viet001.rwl` | ITRDB **viet001**, Vietnam; series `BDFxx`; 1640–2000s. | Genuinely malformed as archived: the series ID `BDF02A` is used by two different cores. Strict reads must refuse and name it; salvage renames the second block `BDF02A2`. |
| `akfirmc.rwl` | Alaska collection (series `FAD`/`FAL`/`FBD`/`FBL`/`FDD`/`FDL`..., 1481–2000s); project-internal copy, not an ITRDB site code. | Contains a hidden self-overlap (an opening partial-decade row with one value too many) that the reader must report as the series overlapping itself. |
| `TMScombined.rwl` | Project-internal: several sites (`MTS`, `TMS`, `TWM` prefixes) combined into one file, 1600s–2020. | Header-less; used for multi-site/combined-file reading. |
| `wwr.rwl` | Project-internal ("weird rwl"): series `WWRC501`, `WWRL801`, 1000–1189. | Irregular layout for the read/write round-trip of awkward files. |
| `xDateRtest.rwl` | Synthetic/anonymised crossdating test set (series `ABC101`–`ABC120`, 1410–). | Used for `xdate` behaviour that needs known mis-dated series. |
| `min.rwl` | Synthetic. Two one-line series (`SeriesA` ending in `999`, `SeriesB` ending in `-9999`). | Minimal file covering both stop markers / precisions. |

`rwl/README.md` is a pointer to this file.

## `csv/` — comma-separated ring widths and reference outputs

| File | Provenance | Notes |
|---|---|---|
| `ca533.csv`, `co021.csv`, `wa082.csv` | CSV versions of the Tucson files above (converted from the ITRDB `.rwl` files with the DPL subprogram YUX). | Year-indexed, one column per series; the `ca533_rwl`/`co021_rwl` fixtures read these. |
| `gp_rwl.csv` | dplR's `gp.rwl` dataset: Gus Pearson Natural Area, Arizona, USA; ponderosa pine (*Pinus ponderosa*); 58 series, 1570–1990. | Written from R. |
| `gp_po.csv` | dplR's `gp.po`: pith offsets for `gp_rwl.csv` in the **dplR convention** (cambial age of the first ring, 1 = pith present). | The RCS tests subtract 1 to obtain dplPy's true offset. |
| `ca533_adsR_Pos.csv`, `ca533_adsR_noPos.csv` | Reference output of dplR's `ads()` (age-dependent spline) for a ca533 series, with `pos.slope = TRUE` / `FALSE`. | R `write.csv` format (`"x"` column). Used to check `ads`/`AgeDepSpline` against dplR numerically. |
| `TTHH_raw.csv` | CSV of `txt/TTHH_raw.txt`. | |

## `txt/` — awkward Tucson files

| File | Provenance | Notes |
|---|---|---|
| `TTHH_raw.txt` | Twisted Tree / Heartrot Hill, Yukon, Canada; white spruce; raw ring widths from E. Cook. | A Tucson file with a one-line free-text header and irregular spacing that is difficult to read properly. |
| `test1.txt` | Synthetic. A three-line Tucson header (`107 1 TUCSON ...`) with a handful of rows. | Header/metadata parsing edge cases. |

## Adding a file

Keep files small (the whole directory is shipped with every checkout), prefer an
ITRDB site code as the file name, and add a row here saying where it came from
and which behaviour it exercises. A file that is intentionally broken should say
so in its name (`*_error.rwl`) and in this table.
