import dplpy as dpl

def test_summary_methods():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    dpl.summary(data)
    dpl.report(data)
    dpl.stats(data)