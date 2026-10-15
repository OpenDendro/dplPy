import warnings

import matplotlib
matplotlib.use("Agg")                                     # headless: no display
import matplotlib.pyplot as plt

import dplpy as dpl

# Headless smoke test: every plot type must draw a figure without error on both
# a .csv and a (header) .rwl dataset. (Content of the figures is checked in the
# unit tests; here we just exercise both file formats end to end.)


def test_all_plot_types_run(ca533_rwl, DATA):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ca667 = dpl.readers(str(DATA / "rwl" / "ca667.rwl"), header=True)
    for data in (ca533_rwl, ca667):
        for t in ("line", "spag", "seg"):
            plt.close("all")
            fig, ax = dpl.plot(data, type=t)
            assert isinstance(fig, plt.Figure), t
            assert len(ax.lines) > 0, t                   # something was drawn
            # under Agg, plot() closes what it "showed": no leaked figures
            assert plt.get_fignums() == [], t
    plt.close("all")
