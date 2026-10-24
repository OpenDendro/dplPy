# Reconstruction validation

A climate reconstruction is a *model* — tree-ring predictors mapped onto a
climate target over a period where both are known, then extrapolated to centuries
where only the tree rings exist. The question that matters is not whether the
model fits the years it was built on, but whether it has genuine skill on years it
has never seen. This page explains how `pcreg` answers that question, what each
statistic means, and where dplPy follows or departs from the method in Ed Cook's
OSR program (Cook, Briffa & Jones 1994).

## A note on the word

dplPy says **validation**, not "verification". The distinction follows Oreskes,
Shrader-Frechette & Belitz (1994): to *verify* (Latin *verus*, true) is to
establish truth, which is possible only in a closed system. A reconstruction is a
model of an open natural system, so its truth can never be established — only its
*legitimacy*. "Validation" denotes exactly that, and it is the honest word for
what these statistics do. dplPy's API reflects this: the withheld window is the
`validation=` argument, and the result carries `.validation`. (Cook's Fortran
subroutine is named `verify`; references to that routine keep its original name.)

## Split-sample validation

The design is simple and strict. The overlap between the predictors and the
climate target is divided into a **calibration** period, where the model is built
(screen → PCA → regression → back-transform), and a **validation** period of
withheld years the model never saw during fitting. Skill is judged only on the
withheld years.

In dplPy the two windows are stated explicitly:

```python
import dplpy as dpl
res = dpl.pcreg(predictors, target,
                calibration=(1965, 1989),
                validation=(1959, 1964))
```

Cook's Fortran *derives* the validation period automatically as every overlap
year outside the calibration window (a block before and a block after). dplPy
deliberately requires the analyst to state it instead. The reason is
data quality: an instrumental target may be unreliable in part of its record — a
gridded PDSI that relaxes toward climatology before the dense-station era, say —
and blindly validating against those years would inflate or wreck the skill for
reasons that have nothing to do with the reconstruction. An explicit window lets
a knowledgeable analyst fence off bad data; it is a control worth keeping.

## Calibration-period skill is not validation

The calibration R² (`result.cal_r2`) — the fraction of target variance the model
explains *in sample* — is reported, but it is a floor, not a verdict. A model with
many predictors screened over few years can fit the calibration period well and
still have no out-of-sample skill. The validation statistics exist precisely
because calibration fit can mislead.

## The validation statistics

All of the following are computed on the withheld years, in `result.stats`, for
both the raw series and (see below) its first differences.

**Reduction of Error (RE)** is the central statistic. It compares the
reconstruction's error against the error of simply guessing the *calibration*
mean every year:

```text
RE = 1 − Σ(observed − reconstructed)² / Σ(observed − calibration_mean)²
```

RE runs from −∞ to 1. Anything above **0** means the reconstruction beats
climatology on the withheld years. It is a *sensitive* test — a low bar, and the
usual minimum threshold for claiming any skill.

**Coefficient of Efficiency (CE)** is the same ratio with a stricter baseline: the
*validation-period* mean rather than the calibration mean.

```text
CE = 1 − Σ(observed − reconstructed)² / Σ(observed − validation_mean)²
```

Because the validation-period mean is the best constant predictor *for those
years*, CE is harder to pass, and $\mathrm{CE} \le \mathrm{RE}$ in general. The gap
between them reflects how much the calibration and validation means differ; a
positive CE is the stronger claim of skill.

**Correlations.** Pearson `r` measures linear agreement; **Spearman** and
**Kendall** are rank-based and insensitive to a few outliers; the **robust**
correlation is the biweight midcorrelation, which down-weights outliers directly.
Each is reported with a one-sided significance (`_p`) — a valid reconstruction is
expected to correlate *positively*, so the test is one-tailed.

**Sign test** counts how many withheld years place the actual and the
reconstruction on the same side of the mean, against a 50/50 coin — pure
directional agreement. The **product-means test** (Fritts) is a weighted version,
asking whether the same-sign agreements dominate. The **equality-of-means test** is
a Welch *t*-test that the reconstruction is not *biased* — that its mean over the
withheld years matches the actual mean.

## High-frequency skill: the first-difference pass

Every statistic is computed a second time on the **year-to-year differences** of
both series (the `_diff` keys in `result.stats`). This is the stricter view. A
reconstruction can look excellent on levels purely by sharing a low-frequency
trend with the target, while having no real interannual skill. Differencing
removes the trend and asks whether the reconstruction tracks the wiggles. Strong
level agreement with weak differenced agreement is a warning that the skill is all
in the trend.

## Cross-validation

Split-sample validation uses one withheld window. Cross-validation
(`cv=`) instead rotates the held-out years through the *calibration* period: with
`cv=1` it leaves out one year at a time (leave-one-out), with `cv=k` it leaves out
contiguous `k`-year blocks, and — crucially — it **refits the entire pipeline**
(screening, PCA, regression) on the remaining years each time before predicting
the held-out ones. The pooled held-out predictions give `cv_r`, `cv_RE` and
`cv_CE`. Because nothing from a held-out year informs its own prediction,
cross-validation exposes overfitting that the in-sample R² hides: a model that
screens hundreds of predictors over a few decades can show a high calibration R²
and a negative `cv_RE`.

## What dplPy implements, and where it departs from Cook's OSR

dplPy reproduces Cook's validation design — split-sample, the full statistical
panel, RE against the calibration mean and CE against the validation mean, and the
first-difference pass — and it was checked number-for-number against the compiled
OSR Fortran on a single-target fixture. It then makes a few deliberate,
documented departures, all in the direction of correctness over exact
reproduction (see [Fidelity & departures](fidelity.md) for the package-wide
principle):

| Area | dplPy | Cook's OSR | Why |
|---|---|---|---|
| **Significance** | SciPy's exact distributions | Wilson-Hilferty / normal approximations | The approximations are era-driven and visibly off at low degrees of freedom; a close-call predictor can land the other side of a screening threshold, but exact reproduction of a 1990s approximation is not the goal. |
| **Robust correlation** | Biweight midcorrelation (Wilcox; = astropy's) | Ed's sum/difference biweight construction | The biweight midcorrelation is the documented, standard robust correlation. |
| **Validation window** | Explicit `validation=` | Auto-derived from the calibration/data overlap | Analyst control over data quality (above). |
| **Cross-validation** | Leave-*k*-out refitting the whole pipeline | A 301-sample bootstrap of the final pairs | A refit CV is the rigorous out-of-sample test; the bootstrap resamples the fitted pairs only. |
| **Precision** | float64 throughout | `real*4` | dplPy never down-casts to reproduce single-precision rounding. |

Two features of the Fortran are intentionally **not** ported: the automatic
two-block (pre- and post-calibration) validation split — a single explicit window
plus cross-validation covers the practical need — and canonical regression, which
is deferred.

## Where to look

The [`pcreg`](../reference/climate.md) API reference documents every statistic in
`result.stats`. The method and the works behind these statistics are collected in
[References](references.md).
