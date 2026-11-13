"""Fire-interval analysis for fire-history composites.

A port of burnr's ``intervals`` (Malevich, Guiterman & Margolis; Malevich et al.
2018, Dendrochronologia, doi:10.1016/j.dendro.2018.02.005). Given a one-series
fire ``composite`` (see :func:`dplpy.composite`), :func:`intervals` computes the
gaps between successive fire years, fits a Weibull or lognormal distribution by
maximum likelihood, and runs Kolmogorov-Smirnov and Shapiro-Wilk goodness-of-fit
tests. It returns a :class:`FhxIntervals` result with the observed summary
statistics and fit quantiles (the lower/median/upper exceedance intervals).

The Weibull fit solves the MLE shape score equation directly (matching R's
MASS::fitdistr to ~1e-6); the lognormal fit is the closed-form MLE. Quantiles and
the KS test use :mod:`scipy.stats`. The KS test statistic matches R's; its
p-value can differ slightly because R and SciPy break ties differently.
"""
import warnings

import numpy as np
from scipy import optimize, stats

from .fhx import _require_fhx, fhx_series_names, fhx_get_events


def _weibull_mle(x):
    """Maximum-likelihood Weibull ``(shape, scale)`` for positive data ``x``,
    by solving the shape score equation (as R's MASS::fitdistr does).

    The shape score is scale-invariant, so it is solved on data normalized to a
    maximum of 1 (``u = x / max(x)``); then every ``u ** k`` stays in ``[0, 1]``
    and large shapes underflow harmlessly instead of overflowing. The scale is
    recovered on the original data as ``max(x) * mean(u ** k) ** (1/k)``.
    """
    x = np.asarray(x, dtype=float)
    xmax = x.max()
    u = x / xmax
    lnu = np.log(u)
    mean_lnu = lnu.mean()

    def score(k):
        uk = u ** k
        return np.sum(uk * lnu) / np.sum(uk) - 1.0 / k - mean_lnu

    # the score rises monotonically from -inf (k -> 0) to a positive limit, so
    # it has a single root in this bracket
    k = optimize.brentq(score, 1e-6, 1e4, xtol=1e-12, rtol=1e-14)
    scale = xmax * np.mean(u ** k) ** (1.0 / k)
    return float(k), float(scale)


def _lognormal_mle(x):
    """Closed-form maximum-likelihood lognormal ``(meanlog, sdlog)`` for ``x``."""
    lnx = np.log(np.asarray(x, dtype=float))
    meanlog = float(lnx.mean())
    sdlog = float(np.sqrt(np.mean((lnx - meanlog) ** 2)))
    return meanlog, sdlog


class FhxIntervals:
    """Result of a fire-interval analysis (see :func:`intervals`).

    Attributes
    ----------
    intervals : np.ndarray
        the observed fire intervals (gaps, in years, between successive fire
        years in the composite).
    densfun : str
        the fitted distribution, ``"weibull"`` or ``"lognormal"``.
    fit : dict
        the MLE parameters -- ``shape``/``scale`` for Weibull, ``meanlog``/
        ``sdlog`` for lognormal.
    kstest : dict
        one-sample Kolmogorov-Smirnov goodness-of-fit against the fitted
        distribution, ``{"statistic", "pvalue"}``.
    shapirotest : dict
        Shapiro-Wilk normality test of the intervals, ``{"statistic",
        "pvalue"}``.
    comp_name : str
        the composite series name.
    event_range : tuple
        ``(min, max)`` year of the events used.
    weibull_median : float
        the Weibull-fit median interval (always a Weibull fit, as in burnr's
        ``print``), ``scale * ln(2) ** (1/shape)``.
    """

    def __init__(self, intervals, densfun, fit, kstest, shapirotest,
                 comp_name, event_range):
        self.intervals = np.asarray(intervals, dtype="int64")
        self.densfun = densfun
        self.fit = fit
        self.kstest = kstest
        self.shapirotest = shapirotest
        self.comp_name = comp_name
        self.event_range = event_range
        # burnr's print always reports a Weibull median, even for a lognormal fit
        wshape, wscale = _weibull_mle(self.intervals)
        self.weibull_median = float(wscale * np.log(2) ** (1.0 / wshape))

    # observed-interval statistics (burnr's mean/median/min/max.intervals)
    def mean(self):
        """Arithmetic mean of the observed intervals."""
        return float(np.mean(self.intervals))

    def median(self):
        """Median of the observed intervals."""
        return float(np.median(self.intervals))

    def min(self):
        """Shortest observed interval."""
        return int(np.min(self.intervals))

    def max(self):
        """Longest observed interval."""
        return int(np.max(self.intervals))

    def std(self):
        """Sample standard deviation of the observed intervals (``ddof=1``)."""
        return float(np.std(self.intervals, ddof=1))

    def quantile(self, q=(0.125, 0.5, 0.875)):
        """Quantiles of the *fitted* distribution at probabilities ``q``.

        The defaults 12.5/50/87.5% are the lower/median/upper exceedance
        intervals reported by burnr.
        """
        q = np.atleast_1d(np.asarray(q, dtype=float))
        if self.densfun == "weibull":
            vals = stats.weibull_min.ppf(q, self.fit["shape"],
                                         scale=self.fit["scale"])
        else:
            vals = stats.lognorm.ppf(q, self.fit["sdlog"],
                                     scale=np.exp(self.fit["meanlog"]))
        return vals if vals.size > 1 else float(vals[0])

    def summary(self):
        """A dict of the common fire-interval summary statistics (the numbers
        burnr's ``print.intervals`` reports)."""
        lo, med, hi = np.atleast_1d(self.quantile())
        return {
            "comp_name": self.comp_name,
            "event_range": self.event_range,
            "total_intervals": int(len(self.intervals)),
            "mean_interval": round(self.mean(), 1),
            "median_interval": round(self.median(), 1),
            "weibull_median": round(self.weibull_median, 1),
            "standard_deviation": round(self.std(), 1),
            "minimum_interval": self.min(),
            "maximum_interval": self.max(),
            "densfun": self.densfun,
            "fit": dict(self.fit),
            "percentiles": {"lower_12.5": round(float(lo), 1),
                            "median_50.0": round(float(med), 1),
                            "upper_87.5": round(float(hi), 1)},
            "ks_statistic": self.kstest["statistic"],
            "ks_pvalue": self.kstest["pvalue"],
            "shapiro_statistic": self.shapirotest["statistic"],
            "shapiro_pvalue": self.shapirotest["pvalue"],
        }

    def plot(self, binwidth=None, ax=None, show=True):
        """Histogram (with a rug) of the observed intervals. Returns
        ``(fig, ax)``; see :func:`dplpy.fhx_demograph` for the plotting
        convention."""
        from .fhx_plot import _plot_intervals_dist
        return _plot_intervals_dist(self, binwidth=binwidth, ax=ax, show=show)

    def __repr__(self):
        s = self.summary()
        if self.densfun == "weibull":
            fitstr = "shape=%.4f scale=%.4f" % (self.fit["shape"],
                                                self.fit["scale"])
        else:
            fitstr = "meanlog=%.4f sdlog=%.4f" % (self.fit["meanlog"],
                                                  self.fit["sdlog"])
        p = s["percentiles"]
        return (
            "Fire interval analysis\n"
            "  composite: %s   events: %d to %d\n"
            "  intervals: %d   mean: %.1f   median: %.1f   Weibull median: %.1f\n"
            "  sd: %.1f   min: %d   max: %d\n"
            "  fit (%s): %s\n"
            "  percentiles  lower(12.5%%): %.1f | median(50%%): %.1f | "
            "upper(87.5%%): %.1f\n"
            "  KS D=%.5f p=%.5f   Shapiro W=%.5f p=%.5f"
            % (s["comp_name"], s["event_range"][0], s["event_range"][1],
               s["total_intervals"], s["mean_interval"], s["median_interval"],
               s["weibull_median"], s["standard_deviation"],
               s["minimum_interval"], s["maximum_interval"],
               self.densfun, fitstr,
               p["lower_12.5"], p["median_50.0"], p["upper_87.5"],
               s["ks_statistic"], s["ks_pvalue"],
               s["shapiro_statistic"], s["shapiro_pvalue"]))


def fhx_intervals(comp, densfun="weibull"):
    """Fire-interval analysis of a one-series fire ``composite``.

    Parameters
    ----------
    comp : fhx
        a fire composite (one series), usually from :func:`dplpy.composite`.
    densfun : str
        distribution to fit, ``"weibull"`` (default) or ``"lognormal"``.

    Returns
    -------
    FhxIntervals
        the observed intervals, the fitted distribution, goodness-of-fit tests,
        and summary/quantile helpers.

    Raises
    ------
    ValueError
        if ``comp`` has more than one series, ``densfun`` is unknown, or there
        are fewer than two intervals (three fire years) to analyse.
    """
    _require_fhx(comp)
    if densfun not in ("weibull", "lognormal"):
        raise ValueError("densfun must be 'weibull' or 'lognormal'.")
    names = fhx_series_names(comp)
    if len(names) > 1:
        raise ValueError("found multiple series in comp; there can only be one.")

    ev = fhx_get_events(comp)
    event_years = np.sort(np.asarray(ev[names[0]], dtype="int64"))
    iv = np.diff(event_years)
    if len(iv) < 2:
        raise ValueError("too few fire events to compute intervals.")

    if densfun == "weibull":
        shape, scale = _weibull_mle(iv)
        fit = {"shape": shape, "scale": scale}
        cdf_args = (shape, 0.0, scale)
        dist = "weibull_min"
    else:
        meanlog, sdlog = _lognormal_mle(iv)
        fit = {"meanlog": meanlog, "sdlog": sdlog}
        cdf_args = (sdlog, 0.0, np.exp(meanlog))
        dist = "lognorm"

    with warnings.catch_warnings():       # KS warns on ties; expected here
        warnings.simplefilter("ignore")
        ks = stats.kstest(iv.astype(float), dist, args=cdf_args)
        sw = stats.shapiro(iv.astype(float))

    all_events = np.concatenate([np.asarray(v, dtype="int64")
                                 for v in ev.values()]) if ev else event_years
    event_range = (int(all_events.min()), int(all_events.max()))

    return FhxIntervals(
        intervals=iv,
        densfun=densfun,
        fit=fit,
        kstest={"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)},
        shapirotest={"statistic": float(sw.statistic),
                     "pvalue": float(sw.pvalue)},
        comp_name=names[0],
        event_range=event_range,
    )
