import matplotlib
matplotlib.use("Agg")                                     # headless: no display
import matplotlib.pyplot as plt

import pytest
import dplpy as dpl


def _ca533():
    return dpl.readers("./tests/data/csv/ca533.csv")


def test_plot_types_create_figures():
    data = _ca533()
    for t in ("line", "spag", "seg"):
        plt.close("all")
        dpl.plot(data, type=t)
        assert len(plt.get_fignums()) >= 1, t             # a figure was drawn
    plt.close("all")


def test_plot_from_filepath_spag_and_seg():
    # ST5 regression: dpl.plot(<path>, type="spag"/"seg") used to pass the raw
    # filepath string to spag_plot/seg_plot and crash; it must now coerce first.
    for t in ("spag", "seg"):
        plt.close("all")
        dpl.plot("./tests/data/csv/ca533.csv", type=t)
        assert len(plt.get_fignums()) >= 1, t
    plt.close("all")


def test_plot_bad_type_raises():
    with pytest.raises(ValueError):
        dpl.plot(_ca533(), type="bogus")
