import pytest
import dplpy as dpl

def test_chron_no_prewhiten_no_biweight():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    res = dpl.chron(data, biweight=False, prewhiten=False, plot=False)
    # TODO: assert contents of res


def test_chron_prewhiten_no_biweight():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    res = dpl.chron(data, biweight=True, prewhiten=False, plot=False)
    # TODO: assert contents of res

def test_chron_prewhiten_with_biweight():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    res = dpl.chron(data, biweight=True, prewhiten=True, plot=False)
    # TODO: assert contents of res


# def test_chron_prewhiten_biweight_plot():
#     data = dpl.readers("./integs/data/csv/ca533.csv")

#     res = dpl.chron(data, biweight=True, prewhiten=True, plot=True)
#     # TODO: assert contents of res

