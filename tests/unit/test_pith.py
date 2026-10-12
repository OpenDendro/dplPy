import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _gp_po():
    return pd.read_csv("tests/data/csv/gp_po.csv")


def test_po_to_wc_offset_convention():
    # dplPy 0.8: pith_offset is a TRUE offset, so n.missing.heartwood equals it
    # (no -1, unlike dplR). Indexed by series.
    po = _gp_po()
    wc = dpl.po_to_wc(po)
    assert list(wc.columns) == ["n_missing_heartwood"]
    assert list(wc.index[:3]) == ["01A", "01B", "03A"]
    assert np.array_equal(wc["n_missing_heartwood"].to_numpy().astype(int),
                          po["pith_offset"].to_numpy().astype(int))


def test_pith_roundtrip():
    # wc_to_po and po_to_wc remain exact inverses under the offset convention:
    # wc_to_po(po_to_wc(po)) == po.
    po = _gp_po()
    back = dpl.wc_to_po(dpl.po_to_wc(po))
    assert np.array_equal(back["series"].to_numpy().astype(str),
                          po["series"].to_numpy().astype(str))
    assert np.array_equal(back["pith_offset"].to_numpy().astype(int),
                          po["pith_offset"].to_numpy().astype(int))


def test_wc_to_po_partial_and_na():
    # dplPy 0.8: pith_offset = missing + unmeasured (NA-aware, no +1); a row with
    # neither a known pith presence nor a missing-heartwood count yields NA.
    wc = pd.DataFrame(
        {"n_missing_heartwood": [5, np.nan, 10],
         "n_unmeasured_inner": [2, 3, np.nan],
         "pith_presence": ["complete", np.nan, "incomplete"]},
        index=["a", "b", "c"])
    out = dpl.wc_to_po(wc)
    assert out.loc[out["series"] == "a", "pith_offset"].iloc[0] == 7      # 5 + 2
    assert pd.isna(out.loc[out["series"] == "b", "pith_offset"].iloc[0])  # unknown
    assert out.loc[out["series"] == "c", "pith_offset"].iloc[0] == 10     # 10 + 0


def test_pith_bad_input():
    with pytest.raises(TypeError):
        dpl.po_to_wc("not a frame")
    with pytest.raises(TypeError):
        dpl.wc_to_po([1, 2, 3])
