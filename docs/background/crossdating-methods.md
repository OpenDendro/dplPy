# Crossdating: dplR vs COFECHA

dplPy's `xdate` can run in two modes that emulate two different, long-established
crossdating programs. They are not variations on one method — they are two
distinct method families, and dplPy reproduces each within its own family. This
page explains the difference and shows how closely dplPy tracks each reference.

## Two method families

| | Default (`preset=None`) | `preset="COFECHA"` |
|---|---|---|
| Emulates | dplR `corr.rwl.seg` | the COFECHA program |
| Correlation | Spearman (rank) | Pearson |
| Prewhitening | Yule-Walker AR | Burg (maximum-entropy) AR |
| Master series | biweight, leave-one-out | arithmetic, z-scored, leave-one-out |
| Detrending of series | as supplied (dplR-style) | spline variance stabilization |
| Critical value | p-value threshold (`p_val`) | COFECHA's t-based critical-r table |
| Segments | dplR bins | COFECHA segment anchoring |
| A flag (weak correlation) | yes (reproduces dplR) | yes |
| B flag (lag shift) | yes (COFECHA-style) | yes |

The default is the right choice when you want dplR-comparable results; the COFECHA
preset is for reproducing (or submitting alongside) a COFECHA run. The
[Crossdating & COFECHA guide](../guide/crossdating.md) shows how to call each.

## How closely does each track its reference?

A useful test is to run one collection through all the relevant programs. On
`wa082` (Hurricane Ridge, *Abies amabilis*, a public ITRDB collection), the
series intercorrelation lands as follows:

| Run | Series intercorrelation |
|---|---|
| dplR `corr.rwl.seg` | 0.564 |
| **dplPy `preset=None`** | **0.571** |
| COFECHA (1996, ITRDB-posted) | 0.586 |
| COFECHA (2026, run locally) | 0.600 |
| **dplPy `preset="COFECHA"`** | **0.599** |

Two things stand out. First, dplPy sits inside each family: `preset=None`
reproduces dplR (0.571 vs 0.564, the small gap a detrending detail), and
`preset="COFECHA"` reproduces the modern COFECHA run (0.599 vs 0.600), matching
per-series "correlation with master" to a mean absolute difference of about
**0.004**. Second — and importantly — **the two authoritative COFECHA runs
themselves disagree** (0.586 vs 0.600), because they interpret one series' data
marker differently. dplPy lands as close to each reference as the references land
to one another.

## "Problem segments" is a screen, not a fixed count

The number of flagged "problem segments" is a threshold-dependent quality screen,
not a fixed property of a collection. Across the five wa082 runs above it ranges
from 0 to 11, driven by the significance threshold, the segment anchoring, and
the flag rules — and the two COFECHA runs alone span 8 to 11. Read the flag count
as "where should I look?", not as a score. The A flag marks segments whose
correlation with the master is weak; the B flag marks segments that correlate
better at a shifted position (a possible dating error). Both point you at
segments to inspect by eye.

## Why not bit-identical?

The residual differences from a given COFECHA binary come down to (a) how a
particular program version reads non-standard data markers, and (b) small
numerical choices in the AR and spline steps — not the crossdating mathematics,
which reproduces per series to within a few thousandths of a correlation unit.
See [Fidelity & departures](fidelity.md) for the project's general stance on
reproduction versus bit-identity, and [References](references.md) for the sources
behind each method.
