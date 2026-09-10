# Variance stabilization

The variance of a mean-value chronology changes through time for reasons that
have nothing to do with the climate or environmental signal it is meant to
carry. As sample depth grows and the mean interseries correlation shifts, the
variance of the average inflates or deflates as a pure statistical artifact —
early, thinly-replicated years look noisier than late, well-replicated ones.
Variance stabilization removes that artifact so the chronology's variance is
comparable across its whole length.

dplPy follows the ARSTAN organization: variance stabilization is a step you
apply to a **finished** chronology, not something welded to how the chronology is
built. That lets you stabilize *any* chronology — standard, residual, or
ARSTAN — with the same engine.

## Two method families

Two published families of methods address chronology variance, and they fix
*different* problems. Choosing between them means diagnosing what is driving the
variance change (Frank et al. 2006).

### rbar / effective sample size (theory-based)

The variance of a mean of correlated series scales as `1/N_eff`, where the
effective (independent) sample size is

$$
N_{\text{eff}} = \frac{n}{1 + (n - 1)\,\bar{r}}
$$

with `n` the sample depth and `r̄` the mean interseries correlation (Osborn et
al. 1997). Stabilization scales each year's departure from the mean by
`sqrt(N_eff)`, which flattens the sample-size / correlation-driven variance
trend while leaving the real common signal intact. This is the method behind
ARSTAN's `stabbm` and Frank et al. (2006). It needs the source RWI matrix (to
get `r̄` and sample depth), or a precomputed `r̄` and depth.

There are two sub-variants:

- **Running rbar** (`rbar_mode="running"`, the default) — `r̄` is recomputed in
  moving windows (Frank's "RUNNINGr"). This is needed when `r̄` itself trends
  through time, and is the widely accepted community default.
- **Constant rbar** (`rbar_mode="constant"`) — a single time-independent `r̄`
  (ARSTAN's Briffa method; Frank's "MEANr"). Appropriate when `r̄` is stable and
  the running estimate would only add noise.

### Spline envelope (ad hoc)

ARSTAN's `stabit` fits a spline to the chronology's **absolute departures**,
divides by that envelope, and restandardizes. It needs **only the chronology** —
no matrix, no `r̄`. Because it is empirical, it flattens *all* time-varying
variance whatever the cause, including a non-stationary component that the rbar
family cannot touch (Frank et al. 2006 found exactly such residual variance and
suggested the spline as the complement). The caveat (Osborn et al. 1997) is that,
being ad hoc, it can also remove *real* low-frequency variance, so it is not
theoretically clean.

### Both, in sequence

`method="both"` applies the rbar correction and **then** a 67 % spline — the
combination ARSTAN offers as menu option `isb=2` and that Frank et al. (2006)
recommend when both a sample-size artifact and a residual non-stationary
component are present.

## `stabilize_chron`: stabilize any chronology

`stabilize_chron` takes a chronology you already built with
[`chron`](chronologies.md) or [`chron_ars`](chronologies.md) and returns the
stabilized version. The `method` is explicit and required, because the families
do different things:

```python
import dplpy as dpl

rwi = dpl.detrend(dpl.readers("ca533.rwl", header=True), fit="Spline")
crn = dpl.chron(rwi)                       # a standard chronology

# rbar / N_eff scaling with the running rbar of the source matrix
vsc = dpl.stabilize_chron(crn, "rbar", rwi=rwi)
vsc.tail(3)
```

```text
           vsc  samp_depth
Year
1981  1.238...          21
1982  1.331...          21
1983  1.297...          21
```

The stabilized chronology comes back in a `vsc` column (plus `samp_depth` when
known). Key arguments:

- `method` — `"rbar"`, `"spline"`, or `"both"` (required).
- `rwi` — the source RWI matrix, used to derive `r̄`, the running `r̄`, and sample
  depth for the rbar family. For a residual or ARSTAN chronology, pass the
  **residual** (prewhitened) matrix (see below).
- `rbar` / `running_rbar` / `samp_depth` — precomputed inputs, as an alternative
  to passing `rwi`.
- `rbar_mode` — `"running"` (default) or `"constant"`.
- `spline_period`, `f` — spline stiffness and frequency response for the
  `"spline"` / `"both"` families (`None` = the 67 % spline, ARSTAN's default).
- `column` — which column to stabilize when passing a DataFrame (default
  `"std"`).
- `return_info` — also return a dict of diagnostics (the `r̄` used, `N_eff`, and
  sample depth).

Passing only the chronology (no matrix) is enough for the spline family:

```python
vsc = dpl.stabilize_chron(crn, "spline")   # needs only the chronology
```

## The `stabilize=` flag on `chron` and `chron_ars`

For convenience, both [`chron`](chronologies.md) and
[`chron_ars`](chronologies.md) accept a `stabilize=` argument so you can build
and stabilize in one call. Each base chronology gets a stabilized sibling column
beside it (`std_vsc`, `res_vsc`, `ars_vsc`), mirroring ARSTAN, where the menu
stabilization option applies to every chronology it produces:

```python
# standard + residual chronologies, each with a stabilized version
crn = dpl.chron(rwi, prewhiten=True, stabilize="both")
list(crn.columns)
# ['std', 'std_vsc', 'res', 'res_vsc', 'samp_depth']

# all three ARSTAN chronologies, each stabilized
ars = dpl.chron_ars(rwi, stabilize="both")
list(ars.columns)
# ['std', 'std_vsc', 'res', 'res_vsc', 'ars', 'ars_vsc', 'samp_depth']
```

Pass extra options through `stabilize_kwargs`:

```python
crn = dpl.chron(rwi, stabilize="rbar",
                stabilize_kwargs={"rbar_mode": "constant"})
```

The defaults (running rbar, 67 % spline) match Frank et al. (2006) and ARSTAN.
`stabilize=None` (the default) leaves the output exactly as it was without the
flag.

### Per-chronology rbar

Following ARSTAN's convention, each chronology is stabilized with the rbar of the
matrix it came from: the **standard** chronology uses the rbar of the standard
(detrended) RWI matrix, and the **residual** and **ARSTAN** chronologies use the
rbar of the **residual** (prewhitened) matrix. The `stabilize=` flags handle this
automatically; when calling `stabilize_chron` directly, pass the appropriate
matrix as `rwi=`.

### The ARSTAN chronology is stabilized before re-reddening

ARSTAN stabilizes the residual chronology and *then* re-reddens it, so
`chron_ars(stabilize=...)` does the same: with `ars_method="arstan"` (the
default), the residual-style mean of the prewhitened series is stabilized
**before** the pooled AR persistence is reintroduced. This keeps `ars_vsc`
consistent with the unstabilized `ars` and faithful to ARSTAN. (With
`ars_method="dplr"`, the ARSTAN chronology is already built by re-reddening each
series and averaging, so `ars_vsc` is stabilized post-hoc on the reddened
chronology.) See [Building chronologies](chronologies.md) for the `ars_method`
distinction.

## `chron_stabilized`: the dplR-parity path

[`chron_stabilized`](../reference/chronology.md) is dplPy's port of dplR's
`chron.stabilized`. Unlike `stabilize_chron`, it takes the RWI matrix and builds
its *own* standard mean-value chronology internally, applying the running-rbar /
`N_eff` correction as it goes:

```python
crn = dpl.chron_stabilized(rwi, win_length=50, min_seg_ratio=1/3)
```

It is the right choice when you want exact dplR parity for the standard
chronology. For a residual or ARSTAN chronology, for the spline or "both"
families, or to stabilize a chronology you already built, use `stabilize_chron`
or the `stabilize=` flag.

## A note on rescaling

Both families restandardize the stabilized chronology to the input chronology's
**full-period** mean and standard deviation. This differs from ARSTAN's
`stabbm`, which rescales to a user-chosen reference period. The full-period
convention keeps the stabilized chronology on the same overall scale as the input
while removing the time-varying variance; see
[Fidelity & departures](../background/fidelity.md) for the rationale.

## Choosing a method

Frank et al. (2006) recommend diagnosing the *source* of the variance change
before choosing — inspect the variance of individual, age-aligned series. Use the
running-rbar method for variance driven by changing sample size or interseries
correlation; use the spline for a residual, non-stationary component the rbar
family cannot explain; and use `"both"` when both are present. See the
[Chronology reference](../reference/chronology.md) for full parameter lists and
[References](../background/references.md) for the source literature.
