import warnings
import pandas as pd
import dplpy as dpl
import pytest


def test_read_ids_pattern_letter_cores_and_variable_site():
    # letter cores, and a site code that changes length between the two families
    names = ["ABC001A", "ABC001B", "ABC002A", "ABCD01", "ABCD02"]
    ids = dpl.read_ids(names)
    assert ids.loc["ABC001A", "tree"] == "ABC001"
    assert ids.loc["ABC001B", "tree"] == "ABC001"     # same tree as ABC001A
    assert ids.loc["ABC002A", "tree"] == "ABC002"     # different tree
    assert ids.loc["ABC001A", "core"] == "A"
    assert ids.loc["ABC001B", "core"] == "B"
    assert ids.loc["ABCD01", "tree"] == "ABCD01"      # 4-letter site, no core
    assert ids.loc["ABCD02", "tree"] == "ABCD02"
    assert ids.loc["ABCD01", "core"] == ""


def test_read_ids_output_shape():
    ids = dpl.read_ids(["ABC001A", "ABC002A"])
    assert list(ids.columns) == ["tree", "core"]
    assert ids.index.name == "series"
    assert list(ids.index) == ["ABC001A", "ABC002A"]


def test_read_ids_accepts_dataframe():
    df = pd.DataFrame({"ABC001A": [1.0, 2.0], "ABC001B": [1.0, 2.0]})
    ids = dpl.read_ids(df)
    assert list(ids.index) == ["ABC001A", "ABC001B"]
    assert (ids["tree"] == "ABC001").all()


def test_read_ids_pattern_digit_core_not_grouped_silently():
    # CAM031 / CAM032 are tree 03 cores 1 and 2, but a DIGIT core is invisible
    # to the pattern parser: the whole digit run is read as the tree number, so
    # the names parse cleanly (no warning) but end up as SEPARATE trees. This is
    # the gotcha that requires an stc mask; documenting it here.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ids = dpl.read_ids(["CAM031", "CAM032"])
    assert not any("could not parse" in str(x.message) for x in w)   # no warning
    assert ids.loc["CAM031", "tree"] == "CAM031"
    assert ids.loc["CAM032", "tree"] == "CAM032"
    assert ids.loc["CAM031", "tree"] != ids.loc["CAM032", "tree"]    # not grouped


def test_read_ids_unparseable_names_warn():
    # names that genuinely do not fit the pattern (no leading letters, or with
    # separators) DO warn and fall back to one-core-per-tree
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ids = dpl.read_ids(["5301A", "AB-01-A"])
    assert any("could not parse" in str(x.message) for x in w)
    assert ids.loc["5301A", "tree"] == "5301A"
    assert ids.loc["AB-01-A", "tree"] == "AB-01-A"


def test_read_ids_stc_mask_digit_cores():
    # stc=(3,2,1): site=CAM, tree=2 digits, core=1 digit -> groups the cores
    names = ["CAM011", "CAM031", "CAM032", "CAM041", "CAM042"]
    ids = dpl.read_ids(names, stc=(3, 2, 1))
    assert ids.loc["CAM031", "tree"] == "CAM03"
    assert ids.loc["CAM032", "tree"] == "CAM03"       # same tree
    assert ids.loc["CAM031", "core"] == "1"
    assert ids.loc["CAM032", "core"] == "2"
    assert ids.loc["CAM011", "tree"] == "CAM01"
    assert ids["tree"].nunique() == 3                 # CAM01, CAM03, CAM04


def test_read_ids_stc_two_element_core_is_remainder():
    ids = dpl.read_ids(["ABC001A", "ABC001BB"], stc=(3, 3))
    assert (ids["tree"] == "ABC001").all()
    assert ids.loc["ABC001A", "core"] == "A"
    assert ids.loc["ABC001BB", "core"] == "BB"        # remainder, any length


def test_read_ids_stc_validation():
    with pytest.raises(ValueError):
        dpl.read_ids(["ABC001A"], stc=(3,))            # wrong length
    with pytest.raises(ValueError):
        dpl.read_ids(["ABC001A"], stc=(3, -1, 1))      # negative
    with pytest.raises(ValueError):
        dpl.read_ids(["ABC001A"], stc=(0, 0, 1))       # site+tree < 1


def test_read_ids_feeds_rwi_stats():
    # round-trip: read_ids output is accepted directly by rwi_stats
    import numpy as np
    rng = np.random.RandomState(0)
    base = np.cumsum(rng.randn(60)) * 0.1 + 5
    df = pd.DataFrame(
        {"ABC001A": base + rng.randn(60) * 0.3,
         "ABC001B": base + rng.randn(60) * 0.3,
         "ABC002A": base + rng.randn(60) * 0.3},
        index=pd.Index(range(1900, 1960), name="Year"),
    )
    ids = dpl.read_ids(df)
    res = dpl.rwi_stats(df, ids=ids, corr="Pearson")
    assert res.iloc[0]["n_trees"] == 2                 # ABC001, ABC002
    assert res.iloc[0]["n_wt"] == 1                    # the ABC001A-ABC001B pair
