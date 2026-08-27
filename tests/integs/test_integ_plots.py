import matplotlib
matplotlib.use("Agg")                                     # headless: no display
import matplotlib.pyplot as plt

import dplpy as dpl

# Headless smoke test: every plot type must draw a figure without error on both
# a .csv and a (header) .rwl dataset. (Content of the figures is checked in the
# unit tests; here we just exercise both file formats end to end.)


def _datasets():
    ca533 = dpl.readers("./tests/data/csv/ca533.csv")
    ca667 = dpl.readers("./tests/data/rwl/ca667.rwl", header=True)
    return ca533, ca667


def test_all_plot_types_run():
    for data in _datasets():
        for t in ("line", "spag", "seg"):
            plt.close("all")
            dpl.plot(data, type=t)
            assert len(plt.get_fignums()) >= 1, t
    plt.close("all")
