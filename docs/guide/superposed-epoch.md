# Superposed epoch analysis

Some climate signals are tied not to a date but to an *event*: the cooling after a
large volcanic eruption, a growth response to a fire or a drought. Superposed epoch
analysis (SEA) finds the average response to a list of such events by compositing a
climate series in windows around each event year. dplPy implements SEA with the
**double bootstrap** of Rao et al. (2019), which adds two things a plain composite
lacks: a *response-uncertainty envelope* — how much the composite depends on which
events you happen to include — and a Monte-Carlo null for significance.

It is a port of the MATLAB `sea_dbl` (Kevin Anchukaitis and Mukund Rao), validated
against it: the deterministic kernel and the response envelope match to machine
precision, and the bootstrap null converges to the MATLAB's.

## How it works

Let the window be `W = pre + post + 1` years wide, centred on each event (year 0).

1. **Composite.** For each event, take the window of the series around it and
   subtract that window's **pre-event mean** (the `pre` years before year 0, which
   excludes year 0 itself). Average the normalized windows down their columns — that
   is the superposed-epoch composite.
2. **Response envelope (the outer bootstrap).** Draw many *unique* subsets of the
   events (without replacement within a subset), composite each, and take the
   5th/50th/95th percentiles at each lag. The spread is the response uncertainty:
   if the composite is robust, it barely moves as events come and go.
3. **Null distribution (the inner bootstrap).** Either reshuffle the real event
   windows in autocorrelation-preserving blocks (`boot_method="block"`), which asks
   whether the response is *locked to year 0*, or draw pseudo-event years at random
   from the series (`boot_method="random"`), which asks whether the *amplitude* is
   unusual against random years. The 1/5/10/90/95/99th percentiles of the resulting
   pseudo-composites are the significance thresholds.

## A worked example

Give `sea` a climate series (a pandas Series indexed by year) and the event years:

```python
import dplpy as dpl

# a reconstruction or climate series, and e.g. large eruptions
res = dpl.sea(series, events=[1453, 1601, 1641, 1695, 1783, 1809, 1815, 1835],
              pre=5, post=10, boot_method="block", seed=1)

res.composite      # the all-event superposed-epoch composite (1 x W)
res.envelope       # 5/50/95th percentile response envelope (3 x W)
res.null           # 1/5/10/90/95/99th percentile thresholds, per lag (DataFrame)
res.lags           # window-relative years, -pre .. post
res.plot()         # the figure: envelope, median response, null bounds
```

A response is significant where the composite falls outside the null bounds (below
the 5 %/1 % lines for a cooling). The figure (also drawn by `dpl.sea(..., plot=True)`)
shades the response envelope, draws the median response, and marks the 5/95 %
(dashed) and 1/99 % (dotted) null thresholds.

## Choices you make

**Window.** `pre` and `post` set how many years before and after each event to
composite. The `pre` years are also the normalization baseline, so keep them clear
of the response. Decide your **year-0 convention** — the eruption/forcing year, or a
one-year lag — before you build the event list; it shifts everything.

**The double bootstrap.** `double_bootstrap=True` (the default) draws `n_subsets`
unique event subsets of `n_sample` events each (default about half the events, which
maximizes the number of distinct subsets). If you ask for more subsets than exist —
`n_subsets > C(n_events, n_sample)` — dplPy warns and returns an ordinary SEA rather
than silently reverting. `double_bootstrap=False` gives the plain composite and a
null only.

**The null.** `boot_method="block"` (default) is autocorrelation-aware and tests
*timing*; `boot_method="random"` tests *amplitude*. They answer different questions,
so report which you used. The block method prints and stores the lag-1
autocorrelation (`res.r1`) and block length (`res.block_length`).

**Reproducibility.** The MATLAB sets no random seed, so its envelope and thresholds
drift run to run. dplPy takes a `seed=` for reproducible results; the composite is
deterministic regardless.

## Reading the output

- `res.composite` is the **all-event** mean. In double-bootstrap mode the plotted
  response line is the *median of the subset composites* (`res.envelope[1]`), which
  is close to but not identical to the all-event mean.
- `res.null` is **pointwise** — a percentile at each lag, with no correction for
  testing all `W` lags. A single lag crossing a threshold is weak evidence; a
  sustained, physically-sensible excursion (a cooling at year 0 recovering over a
  few years) is the strong case.
- Events whose window would run past either end of the series are dropped with a
  warning, and `res.events` is the list actually used.

## Fidelity

`sea` reproduces the MATLAB `sea_dbl` exactly on the deterministic kernel (window
extraction, pre-event normalization, the composite, the block length from lag-1
autocorrelation, and the percentiles — using `numpy`'s `method="hazen"` to match
MATLAB's `prctile`) and on the response envelope when the same event subsets are
drawn; the block-bootstrap null converges to the MATLAB's distribution. dplPy also
fixes a few issues in the original (the subset count uses `math.comb` rather than
factorials, the unique-draw loop has a correct escape, and the run is seedable). See
[References](../background/references.md) for the method.
