import dplpy as dpl

# Each xdate() call here does a leave-one-out crossdate, and for every series/bin it
# slides +/-10 years and re-correlates at each lag (dominant cost when show_flags is
# on, which is the default). That cost scales with series count, and each test below
# calls xdate 3x, so trimming to a real subset of the series keeps every code path
# (multiple bin_floor/slide_period values, real messy tree-ring data) exercised while
# cutting runtime roughly in proportion to the series count.
def test_xdate_diff_bins():
    ca533 = dpl.readers("./tests/data/csv/ca533.csv").iloc[:, :15]

    ca533_bindata_1 = dpl.xdate(ca533, bin_floor=0)
    ca533_bindata_2 = dpl.xdate(ca533, bin_floor=10)
    ca533_bindata_3 = dpl.xdate(ca533, bin_floor=100)

def test_xdate_diff_slide_periods():
    ca533 = dpl.readers("./tests/data/csv/ca533.csv").iloc[:, :15]

    ca533_bindata_1 = dpl.xdate(ca533, slide_period=30)
    ca533_bindata_2 = dpl.xdate(ca533, slide_period=50)
    ca533_bindata_3 = dpl.xdate(ca533, slide_period=80)

def test_xdate_diff_corrs():
    # ca667.rwl has 310 series; xdate's leave-one-out crossdating is O(n_series)
    # chron() rebuilds, so the full file takes minutes here just to check that
    # corr="Spearman" vs "Pearson" both run without error. A real subset of the
    # same file exercises identical code paths (real, messy, differing-length
    # tree-ring series) in a few seconds instead.
    ca667 = dpl.readers("./tests/data/rwl/ca667.rwl", header=True).iloc[:, :30]

    ca667_bindata_1 = dpl.xdate(ca667, corr="Spearman")
    ca667_bindata_2 = dpl.xdate(ca667, corr="Pearson")

def test_xdate_not_prewhitened():
    ca667 = dpl.readers("./tests/data/rwl/ca667.rwl", header=True).iloc[:, :30]

    ca667_bindata = dpl.xdate(ca667, prewhiten=False)

# Commented out because plots block execution in vscode. WIP
# def test_xdate_plot():
#     co021 = dpl.readers("./integs/data/rwl/co021.rwl")

#     dpl.xdate_plot(co021)
