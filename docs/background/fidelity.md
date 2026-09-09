# Fidelity & departures

dplPy is a port. Its goal is to reproduce the *methods* of the established
dendrochronology tools — primarily [dplR](https://github.com/OpenDendro/dplR),
and, where relevant, COFECHA, ARSTAN, and CRUST — so that results are comparable
across ecosystems. This page explains what "reproduce" means here, how it is
checked, and the places where dplPy deliberately departs from its references.

## The principle

Two commitments guide the port:

1. **Where dplPy claims to reproduce a reference method, that claim is checked
   against the reference implementation** — not from memory or documentation, but
   against the actual source.
2. **Every deliberate departure is documented** at the point it occurs (in the
   relevant function's docstring) and summarized here.

In 2026 the crossdating and standardization stack was put through a systematic
fidelity audit: every behavioral constant and every "matches dplR / COFECHA /
ARSTAN" claim in the source was verified against the authoritative references —
the dplR 1.8.0 R sources, the CRUST Fortran (Melvin & Briffa), and the COFECHA
Fortran. Unsourced constants were removed or documented, and overstated claims
were corrected.

## A note on precision

Reproducing a *method* is not the same as being bit-identical to a compiled
Fortran program. For pure-Python ports of dplR routines, dplPy follows dplR's
algorithm step for step. For the COFECHA and ARSTAN emulations, dplPy reproduces
the headline numbers closely (see [Crossdating: dplR vs COFECHA](crossdating-methods.md))
but will not match the original binaries to the last digit — the reference
programs themselves differ from run to run and version to version. dplPy's
documentation avoids claiming a numerical precision it cannot demonstrate.

## Documented departures

These are the places where dplPy intentionally differs from dplR (or from the
tool it is emulating). None is accidental; each is chosen and documented.

| Area | Departure | Why |
|---|---|---|
| **Crossdating B flag** | `xdate` flags a segment "B" when its correlation is higher at a non-dated lag. dplR's `corr.rwl.seg` has **no B flag**; this is a COFECHA-style addition, active in both the default and the COFECHA preset. | COFECHA's lag-shift information is useful; it is provided in both modes by design. |
| **Floating-series match** | `xdate_floater` selects the best placement by the highest *t* statistic; dplR's `xdate.floater` uses the highest correlation *r*. dplPy also omits dplR's future-year trim. | *t* accounts for the degrees of freedom at each overlap (Wilson 2026). |
| **RCS / SSF `preset="crust"`** | A minimum curve-value floor of 0.02 ("one ring") is applied. This comes from CRUST (Melvin & Briffa 2014), **not** from dplR's `rcs`/`ssf`. | It reproduces the CRUST behavior the preset is meant to emulate. |
| **`chron_stabilized`** | `min_seg_ratio` defaults to exactly `1/3`, matching dplR's `winLength/3` overlap cutoff. | Exact rather than a rounded 0.33, so the default matches dplR. |
| **`detrend` negative period** | A negative `period` is read as a *percent spline* (`|period|/100 × n`). | The ARSTAN convention (`idt < -9 → years cutoff = idt/100·n`); dplR simply rejects non-positive stiffness. |
| **`writers`** | Interior gaps use a negative sentinel (default `-99`, a dplPy convention), and the default precision is `0.001` (dplR defaults to `0.01`). | dplR has no single interior-gap encoding; the sentinel is explicit and reversible on read. |
| **`readers`** | Anomalous negative values (not the `-9999` stop marker) are set to NaN with a warning rather than silently altered; `*-noaa.rwl` template files are rejected with an informative error. | Ring widths cannot be negative; keeping them out of the data is safer than coercing them. |

## Where to look

Each function's docstring states its reference and any departure; the
[API Reference](../reference/io.md) surfaces those docstrings directly. The
crossdating method families — the largest and most consequential area of
emulation — have their own page: [Crossdating: dplR vs COFECHA](crossdating-methods.md).
The works behind these methods are collected in [References](references.md).
