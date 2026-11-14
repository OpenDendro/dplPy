"""dplPy fire-history (FHX2) namespace.

All fire-history functionality lives under ``dpl.fhx``: reading and writing FHX2
fire-scar files, the tidy event-table model, series and site statistics,
compositing, fire-interval analysis, and the fire chart (demograph). For
example::

    import dplpy as dpl
    x    = dpl.fhx.read("lgr2.fhx")
    comp = dpl.fhx.composite(x)
    iv   = dpl.fhx.intervals(comp)
    dpl.fhx.demograph(x, composite_rug=True)

A port of the R package burnr (Malevich, Guiterman & Margolis; Malevich et al.
2018, Dendrochronologia, doi:10.1016/j.dendro.2018.02.005), validated against it.

Fire years from a one-series composite can also be fed straight to the general
``dpl.sea`` (superposed epoch analysis): ``dpl.sea(climate, dpl.fhx.composite(x))``.
"""
from .fhx_model import read, write, frame, as_frame, is_fhx, events, sort
from .fhx_stats import stats, summary, sample_depth, percent_scarred, composite
from .fhx_intervals import intervals, Intervals
from .fhx_plot import demograph

__all__ = [
    # I/O and the event-table model
    "read", "write", "frame", "as_frame", "is_fhx",
    # accessors
    "events", "sort",
    # statistics and compositing
    "stats", "summary", "sample_depth", "percent_scarred", "composite",
    # fire-interval analysis
    "intervals", "Intervals",
    # plotting
    "demograph",
]
