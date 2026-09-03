import numpy as np
import pandas as pd
import pytest

import dplpy as dpl

PI = np.pi


def _df(data, years):
    return pd.DataFrame(data, index=pd.Index(years, name="Year"))


def test_bai_in_from_pith_hand_computed():
    # rings 1,1,1 from the pith -> radii 1,2,3 -> BAI = pi(R_t^2 - R_{t-1}^2)
    df = _df({"A": [1.0, 1.0, 1.0]}, [2001, 2002, 2003])
    out = dpl.bai_in(df)
    assert np.allclose(out["A"].to_numpy(), [PI, 3 * PI, 5 * PI])


def test_bai_in_with_d2pith_offset():
    # rings 1,1 with a 5-unit distance to pith -> radii 6,7
    df = _df({"A": [1.0, 1.0]}, [1, 2])
    out = dpl.bai_in(df, d2pith={"A": 5.0})
    assert np.allclose(out["A"].to_numpy(), [PI * (36 - 25), PI * (49 - 36)])


def test_bai_out_default_matches_bai_in_when_pith_reached():
    # with no diameter, radius = sum of ring widths, so the outermost ring sits at
    # the full radius and bai_out equals bai_in with d2pith = 0
    df = _df({"A": [1.0, 2.0, 1.5, 3.0]}, [1, 2, 3, 4])
    assert np.allclose(dpl.bai_out(df).to_numpy(), dpl.bai_in(df).to_numpy())


def test_bai_out_with_diameter():
    # rings 1,1; diameter 10 -> radius 5; inner ring pi(16-9)=7pi, outer pi(25-16)=9pi
    df = _df({"A": [1.0, 1.0]}, [1, 2])
    out = dpl.bai_out(df, diam=pd.DataFrame({"series": ["A"], "diam": [10.0]}))
    assert np.allclose(out["A"].to_numpy(), [7 * PI, 9 * PI])


def test_shape_and_nan_preserved():
    df = _df({"A": [1.0, 1.0, np.nan], "B": [np.nan, 2.0, 2.0]}, [1, 2, 3])
    for out in (dpl.bai_in(df), dpl.bai_out(df)):
        assert out.shape == df.shape
        assert list(out.index) == list(df.index)
        assert np.isnan(out.loc[3, "A"]) and np.isnan(out.loc[1, "B"])
    # B (rings 2,2 from pith) -> radii 2,4 -> 4pi, 12pi
    assert np.allclose(dpl.bai_in(df)["B"].dropna().to_numpy(), [4 * PI, 12 * PI])


def test_dict_series_and_df_inputs_agree():
    df = _df({"A": [1.0, 1.0], "B": [2.0, 1.0]}, [1, 2])
    diam = {"A": 6.0, "B": 8.0}
    via_dict = dpl.bai_out(df, diam=diam)
    via_ser = dpl.bai_out(df, diam=pd.Series(diam))
    via_df = dpl.bai_out(df, diam=pd.DataFrame({"series": ["A", "B"], "diam": [6.0, 8.0]}))
    assert via_dict.equals(via_ser) and via_dict.equals(via_df)


def test_missing_series_raises():
    df = _df({"A": [1.0, 1.0], "B": [1.0, 1.0]}, [1, 2])
    with pytest.raises(ValueError):
        dpl.bai_out(df, diam={"A": 5.0})            # B missing
    with pytest.raises(TypeError):
        dpl.bai_in(df, d2pith=5.0)                  # not a mapping/frame


def test_exported():
    assert hasattr(dpl, "bai_out") and hasattr(dpl, "bai_in")
