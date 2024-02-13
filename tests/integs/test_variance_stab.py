import dplpy as dpl

def test_chron_stab_no_biweight():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    res = dpl.chron_stabilized(data, biweight=False)
    # TODO: assert contents of res


def test_chron_stab_with_biweight():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    res = dpl.chron_stabilized(data)
    # TODO: assert contents of res

def test_chron_stab_with_running_rbar():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    res = dpl.chron_stabilized(data, running_rbar=True)
    # TODO: assert contents of res
