# Correlation and response functions

A correlation or response function is the classic monthly screen of a chronology
against climate: one coefficient for each month of one or more climate variables,
running from the previous growing season into the current one, each with a
confidence interval or significance flag, read as a bar chart across the months.
It answers "which months is this site's growth responding to, and how strongly."
`dpl.climate_response` computes it, by either of the two standard methods, with
bootstrapped significance.

The idea goes back to Fritts et al. (1971) and the program DENDROCLIM2002 (Biondi
& Waikul 2004), and the interface here follows the one many users know from the R
package treeclim (Zang & Biondi 2015). dplPy's results match treeclim's to
machine precision (see [Validation](#validation) below).

## Correlation function or response function

Both relate the chronology to the individual months of the climate data; they
differ in how they handle the fact that neighbouring months are themselves
correlated.

The **correlation function** (`method="correlation"`) is the simplest thing: the
Pearson correlation of the chronology with each month separately. It is easy to
read, but because, say, a warm June and a warm July tend to go together, two
adjacent months can both correlate with growth even if only one of them is doing
the work — the correlation function cannot tell them apart.

The **response function** (`method="response"`, the default) is Fritts' answer to
that. Instead of correlating month by month, it regresses the chronology on the
whole monthly-climate matrix at once, so each month's coefficient is its
contribution *after* accounting for the others. A plain multiple regression on
collinear months would be unstable, so the predictors are first rotated to their
principal components (which are uncorrelated by construction), the chronology is
regressed on those, and the coefficients are transformed back to one weight per
month. The number of components kept is set by the PVP criterion (Guiot 1991):
keep the leading components while the running product of their eigenvalues stays
above one. Dropping the trailing components is what regularizes the fit against
the near-collinearity of adjacent months; keeping all of them would just return
the unstable standardized multiple-regression coefficients.

Neither method is partial with respect to a *second climate variable* — if you
want the correlation of growth with precipitation controlling for temperature,
that is what [`seascorr`](seascorr.md) does, by season. And if your question is
whether a single relationship is *stable through time* rather than which months
matter, that is [`moving_corr`](moving-correlation.md). `climate_response` is the
monthly, multi-variable screen; the three functions answer different questions
and share machinery rather than scope.

## The month window

Months use the dendroclimatological sign convention, the same `end_month` anchor
as [`seascorr`](seascorr.md): positive months are the current growth year (`1` =
January … `12` = December) and negative months the previous year (`-1` = January
… `-12` = December), with `-13…-24` reaching into the year before that. The window
is the contiguous span from `start_month` to `end_month`, so the default
`start_month=-6, end_month=9` is previous June through current September, and
`start_month=-9` would start at previous September. Every month in that span, for
every climate variable, becomes one predictor.

## Significance by bootstrap

Rather than analytic standard errors, significance is bootstrapped (Guiot 1991),
with three choices:

- **`boot="exact"`** (the default) holds the climate fixed and simulates the
  chronology 1000 times as a series with the same spectrum — hence the same
  autocorrelation — using the exact method of Percival & Constantine (2006), the
  same engine [`seascorr`](seascorr.md) uses. Each observed coefficient is tested
  against that null distribution by its Weibull non-exceedance probability. This
  path reuses dplPy's already-validated simulation, and is the recommended
  default; it reports a significance flag but no confidence interval.
- **`boot="std"`** is the classic DENDROCLIM2002 bootstrap: resample the years
  with replacement (chronology and climate together), refit, and take the
  percentile confidence interval of the coefficients. A coefficient is
  significant when that interval does not bracket zero (Dixon 2001).
- **`boot="stationary"`** is the same percentile approach but with the stationary
  block bootstrap of Politis & Romano (1994): it resamples blocks of years, so
  the resampled series keeps the autocorrelation that the plain `"std"` bootstrap
  destroys. The block length is chosen automatically by the Politis & White
  (2004) rule and reported in `result.settings["block_length"]`.

`boot="none"` returns point estimates only. The test level is set by `ci` (one of
0.1, 0.05, 0.01).

## A worked example

Give `climate_response` a year-indexed chronology and one or more monthly climate
variables (each in any shape [`seascorr`](seascorr.md) accepts — wide, tidy, or a
datetime-indexed series). A dict keys several variables by name:

```python
import dplpy as dpl

res = dpl.climate_response(crn, {"temperature": temp, "precip": prcp},
                           start_month=-6, end_month=9,
                           method="response", boot="exact", seed=1)

res.table        # one row per (variable, month): coef, ci_lower/upper,
                 # significant, p
print(res.summary())
res.plot()
```

The figure stacks one panel per climate variable, a bar per month across the
window — previous-year months in lower case, current-year months in upper case,
with a divider at the year boundary (the layout of Fritts 1971, Fig. 4).
Significant months are filled, the rest left open, and the resampling bootstraps
add a confidence-interval whisker on each bar. Temperature panels are coloured
red and moisture variables blue, auto-detected from the variable names.

## Choices you make

**Method.** `"response"` (the default) is the multicollinearity-aware choice and
is usually what you want when several months or variables are in play;
`"correlation"` is the simpler per-month screen, and is fine when you only have a
single variable or want the most direct reading.

**Bootstrap.** `"exact"` is the recommended default. Prefer `"stationary"` when
you specifically want resampling-based confidence intervals that respect the
chronology's persistence; `"std"` is the classic iid bootstrap and will tend to
call more months significant precisely because it ignores that persistence.

**The window.** A wider `start_month…end_month` span tests more months but uses
more predictors, so it needs a longer record (the response function needs more
years than predictors, and the exact bootstrap needs at least 31 years).

**Reproducibility.** All three bootstraps are stochastic; pass a `seed=` for
reproducible significance. The point estimates are deterministic.

## Reading the output

The response and correlation functions can disagree, and the difference is
informative: a month that correlates with growth on its own but drops out of the
response function was probably riding on a neighbouring month that carries the
real signal. Read the coefficients together with the significance, and remember
that scanning a dozen-plus months is a multiple-comparisons exercise — a single
month just clearing the line is weak evidence, a physically-sensible run of
months is strong. The response coefficients are standardized (the effect of a
one-standard-deviation change in that month's climate on the chronology), so they
are comparable across months and variables.

## Validation

The coefficients were checked against treeclim 2.0.8 on its own `muc_spruce` /
`muc_clim` data: for both the correlation and the response function, dplPy's
coefficients match treeclim's to about 2.6 × 10⁻¹⁵ — the full chain of
z-scoring, the predictor correlation matrix, the eigendecomposition, PVP
retention, the component regression, and the back-transform reproduces the
reference implementation. The bootstrap confidence intervals and significance
flags are stochastic and cannot be matched bit-for-bit across different random
number streams; they rest on the simulation and Weibull machinery already
validated for [`seascorr`](seascorr.md) against Meko's SEASCORR. See
[References](../background/references.md) for the methods.

## A planned extension

Component retention currently uses the PVP criterion only. Fritts' original
response function selected components by percent of variance explained and used
stepwise regression; offering those is a natural future option, since much of
dplPy's climate work traces back to Fritts.
