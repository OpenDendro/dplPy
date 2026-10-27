# Moving correlation

A climate signal in a tree-ring chronology is often assumed to be constant: we
calibrate over the instrumental period and reconstruct the past on the strength
of that one number. But the correlation between growth and climate can change —
a site can lose (or gain) sensitivity to a season as the climate, the stand, or
the trees themselves change. A *moving correlation* looks for that: it slides a
window along the two series and correlates them inside each window, producing a
trace of the relationship through time.

The trace is easy to compute and easy to over-read. `dpl.moving_corr` is built
around the second problem as much as the first. A moving correlation on
overlapping windows **wobbles even when the true relationship is perfectly
constant**, so the interesting-looking excursions in a raw trace are partly, and
sometimes entirely, an artifact of the method. The function pairs the trace with
significance tests designed for exactly this situation, and a figure that shows
the uncertainty rather than hiding it.

## Three things to keep in mind

**Overlapping windows manufacture wobble (the Slutsky-Yule effect).** Adjacent
windows share all but one or two years, so successive points on the trace are
highly dependent, and a short-window correlation of two persistent (red) series
drifts up and down on its own. Averaging and differencing operations impose
their own low-frequency rhythm on random data — Slutsky and Yule showed this a
century ago — and a moving correlation is such an operation. The consequence: a
smoothly undulating trace is the *expected* behaviour of the null, not evidence
against it. The remedy here is to make the null wobble the same way. The
significance tests compare the observed trace against surrogate series that carry
the same persistence (lag-1 autocorrelation) of both your chronology and your
climate variable, the same background correlation, and the same window and
offset. Whatever wobble overlapping windows induce is present in the null too, so
the test asks only whether the observed trace wobbles *more than that*.

**You are running many tests at once (multiplicity).** A trace of 76 windows is
76 correlations; scanning it for the highest one and testing that is not the same
as testing a single pre-chosen window, and the season grid (several season
lengths × twelve ending months) multiplies the count again. Flagging each window
that individually clears a 95% line guarantees false positives. `moving_corr`
handles the within-trace multiplicity with a *peak* test whose null is the
maximum correlation anywhere in each surrogate trace: a single threshold line
then already accounts for having searched the whole record, so a point above it
is significant at the stated level across the entire trace, not just at one spot.

**"Significant" and "not significant" is not a real boundary (Gelman & Stern).**
The difference between a window that clears the line and one that just misses it
is not itself significant, and a map of significant/not-significant cells invites
you to read a change in the signal where there may only be a change in the
p-value. So `moving_corr` does not shade per-window significance. It shows the
estimate with its uncertainty — the trace, a per-window confidence band, and one
threshold line — and reports the whole-trace tests as single verdicts. The
question it answers is "is this trace, as a whole, more structured than chance,"
not "which individual years are significant."

## How it works

1. **Seasonalize.** The monthly climate variable is collapsed into one seasonal
   series using the same machinery as [`seascorr`](seascorr.md): you give an
   ending month and a season length (e.g. ending month 8, length 3 = Jun–Aug),
   and whether to aggregate by `"mean"` (temperature-like) or `"sum"`
   (precipitation totals).
2. **Slide and correlate.** The chronology and the seasonal series are aligned on
   their common years, and a window of `window` years is stepped along them by
   `offset` years; each window yields one correlation. `offset=1` gives a fully
   overlapping, smooth trace; `offset=window` gives adjacent, non-overlapping
   (independent) windows; values between trade smoothness for independence. The
   point is plotted at the window's centre by default (`position`).
3. **Build the matched null.** For the significance tests, many surrogate pairs
   are generated at the observed background correlation with each series' lag-1
   persistence (the red-noise null; `noise="white"` drops the persistence), and
   the same moving correlation — same window, same offset — is computed for each.
4. **Test the whole trace.** Three global statistics of the observed trace are
   compared against their null distributions: its standard deviation (is the
   relationship *more variable* than chance — the Gershunov et al. 2001 test),
   its maximum (is the strongest window real), and its high-to-low range.

## A worked example

Give `moving_corr` a year-indexed chronology and one monthly climate variable:

```python
import dplpy as dpl

res = dpl.moving_corr(crn, precip, end_month=8, season_length=3, agg="sum",
                      window=30, offset=1, noise="red", seed=1, plot=True)

res.trace        # DataFrame: window position year -> start, end, r, ci_lower/upper
res.tests        # the global tests: "std", "peak", "range"
res.overall_corr # full-period correlation
res.threshold(0.95)        # the upper (peak) significance line
res.threshold_lower(0.95)  # the lower (trough) significance line
print(res.summary())
```

The figure (also drawn by `res.plot()`) shows the correlation trace in blue with
its per-window confidence band, the overall correlation, and the peak/trough
threshold lines; windows that cross a threshold are dotted. The whole-trace test
verdicts are printed in the caption — as statements about the trace, not about
individual windows.

## Choices you make

**Window and offset.** `window` sets the width in years. A longer window (30 is
the default, a climate-normal length) gives a smoother, better-estimated trace
but resolves change more slowly and needs a longer record; a shorter window is
more responsive but noisier, with wide per-window intervals. `offset` is the
overlap control: keep it at 1 for a continuous trace, or raise it toward `window`
when you want windows that are closer to independent. The null is always computed
at the same window and offset, so the test stays fair whichever you choose.

**The null model.** `noise="red"` (the default) matches both series' lag-1
persistence and is the right null for persistent climate and tree-ring series;
`noise="white"` is the classical Gershunov null and will flag more traces as
"unstable" precisely because it ignores the wobble persistence induces.

**The peak test's question.** `peak_condition="observed"` (the default) asks
whether the strongest window is elevated *above the established background
relationship*; `peak_condition="zero"` asks whether any relationship exists at
all. Once you already know two series are related overall, the zero-conditioned
test is easy to pass and says little about the peak being special.

**The confidence band.** `ci="fisher"` (the default) is an analytic Fisher-z
interval at an effective sample size reduced for autocorrelation, so short windows
honestly show wide bands; `ci="bootstrap"` is a moving-block bootstrap;
`ci="none"` omits the band.

**Reproducibility.** The surrogates and the bootstrap are stochastic; pass a
`seed=` for reproducible significance. The trace itself is deterministic.

## Reading the output

A wobbling trace is not, by itself, evidence that the climate signal is unstable —
that is the whole reason the tests exist. Read the trace *together with* the
verdicts: if the `std` test is not significant, the variation you see is within
what red noise and overlapping windows produce, however structured it looks. A
single window crossing a line is weak; a sustained, physically-sensible shift
(for instance a signal that strengthens after a known stand disturbance, or
weakens in a late-century divergence) supported by a significant whole-trace test
is the strong case. The per-window band is a reminder that each point is itself
uncertain, especially for short windows.

## The season grid

Everything above is one chronology against one season. To ask the stability
question across the whole seasonal grid — every season length by every ending
month, as in [`seascorr`](seascorr.md) — pass `season_lengths` (plural) instead of
a single `season_length`:

```python
grid = dpl.moving_corr(crn, precip, end_month=12, season_lengths=(1, 3, 6, 12),
                       window=30, agg="sum", seed=1)

grid.grid        # one row per season: overall_r, stability_p, fdr_sig, ...
grid.cube        # the moving-correlation trace of every season
grid.field       # field significance (see below)
grid.plot(kind="small_multiples", seasons=[(3, 5), (3, 8)])
grid.plot(kind="dominant")
grid.plot(kind="surface")
print(grid.summary())
```

This returns a `MovingCorrGridResult` with three complementary views, none of which
uses a per-cell significance mask:

- **`small_multiples`** (the default) tiles the single-season trace-with-band plot
  across the seasons you pass in `seasons=` (a list of `(length, end_month)`), each
  panel an honest estimate-and-uncertainty display with `*` marking the
  FDR-significant seasons. With no `seasons=` it warns and falls back to the
  strongest-correlating season at each length.
- **`dominant`** shows, for one season length, which ending month carries the
  strongest correlation at each window — the trace of the "winning" season over
  time, coloured by its strength, with a grey bar spanning the seasons within 90 %
  of the best (narrow where the choice is sharp, wide where it is ambiguous). It
  answers "does the dominant season shift," but because it always draws a
  confident-looking line, read it together with the field-significance verdict.
- **`surface`** is the full picture: a season-by-time heatmap of the moving
  correlation on a diverging red/blue scale (red positive, white near zero), so
  you see at a glance which seasons carry the signal and how it changes through
  time — never a grid of significance flags flickering from cell to cell. Pass
  `mode="anomaly"` to map each cell's *departure from its own full-period
  correlation* instead, which flattens the magnitude to emphasise change alone.

### Field significance

Scanning dozens of seasons across many windows is a severe multiple-comparisons
problem, made worse because neighbouring seasons share months and so move
together. dplPy reports two things, for two different questions. The
**Livezey-Chen** field test is the grid-level headline: it counts the
locally-significant cells and compares that count against the distribution
produced by a chronology *unrelated* to climate (reusing the real climate grid, so
the inter-season correlation is preserved), answering "is this chronology's
seasonal signal, as a whole, more non-stationary than noise?" The
**Benjamini-Hochberg FDR** set is the per-season annotation — which individual
seasons survive multiplicity — shown as the `*` marks and the red surface labels.
The point, again in the spirit of Gelman & Stern (2006), is a grid-level verdict
plus honest per-season effect sizes, not a thresholded map that invites you to
read the on/off pattern as the signal.

## Fidelity

The significance engine is a port of the `moving_corr_sig` package (K.
Anchukaitis), validated against Table 1 of Gershunov et al. (2001): for two
white-noise series correlated at 0.30 with an 11-year window, the 95th percentile
of the null standard-deviation distribution is ≈ 0.36. dplPy's port reproduces
that package bit-for-bit given the same random stream — the moving correlation,
the surrogate generation, the AR(1) fit, and the Monte-Carlo ensemble all match —
and the moving correlation itself matches a direct rolling correlation to machine
precision. See [References](../background/references.md) for the methods.
