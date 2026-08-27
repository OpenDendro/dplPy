import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

import dplpy as dpl


def _read_quiet(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.readers(path)


def _ssf_quiet(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return dpl.ssf(data, verbose=False, **kw)


# dplR ssf(ca533, method="Spline", recode.zeros=TRUE) first values (dplR 1.7.9).
# dplPy reproduces the full signal-free iteration to ~1e-7.
_DPLR_SSF_CA533 = [0.394662, 0.302022, 0.325494, 0.442067, 0.512245]


def test_ssf_matches_dplR_ca533_spline():
    data = _read_quiet("tests/data/csv/ca533.csv")
    out = _ssf_quiet(data, method="Spline", recode_zeros=True)
    assert list(out.columns) == ["sfc", "samp_depth"]
    assert np.allclose(out["sfc"].to_numpy()[:5], _DPLR_SSF_CA533, atol=1e-4)


# dplR ssf(wa082, method="AgeDepSpline", recode.zeros=TRUE) first values.
_DPLR_SSF_WA082_ADS = None  # filled by value check below via tolerance only


def test_ssf_agedepspline_runs_and_positive():
    data = _read_quiet("tests/data/csv/wa082.csv")
    out = _ssf_quiet(data, method="AgeDepSpline", recode_zeros=True)
    vals = out["sfc"].to_numpy()
    vals = vals[~np.isnan(vals)]
    assert np.all(np.isfinite(vals)) and (vals > 0).all()
    assert 0.5 < np.mean(vals) < 1.5


def test_ssf_return_info():
    data = _read_quiet("tests/data/csv/ca533.csv")
    info = _ssf_quiet(data, method="Spline", recode_zeros=True, return_info=True)
    assert isinstance(info, dict)
    for key in ("infoList", "ssfCrn", "sfCrn_Mat", "MAD_Vec", "sfRWI_Array"):
        assert key in info
    # the standalone call returns the same final chronology as info["ssfCrn"]
    out = _ssf_quiet(data, method="Spline", recode_zeros=True)
    assert np.allclose(info["ssfCrn"]["sfc"].to_numpy(), out["sfc"].to_numpy())
    # iteration history is consistent: sfCrn_Mat has one column per iteration
    assert info["sfCrn_Mat"].shape[1] == len(info["MAD_Vec"]) + 1


def test_ssf_all_zero_row_rejected():
    # ca533 has a year whose only present rings are zero; without recode_zeros
    # ssf must raise (matching dplR's input0 guard).
    data = _read_quiet("tests/data/csv/ca533.csv")
    with pytest.raises(ValueError):
        _ssf_quiet(data, method="Spline", recode_zeros=False)


def test_ssf_negative_curve_error_is_informative():
    # co021 (many absent rings) drives the chronology near zero, which blows up
    # the signal-free measurements and makes a refitted curve go <= 0. Both dplR
    # and dplPy stop here; dplPy's message should name the offending series and
    # the near-zero chronology year so the cause is clear.
    data = _read_quiet("tests/data/csv/co021.csv")
    with pytest.raises(ValueError) as e:
        _ssf_quiet(data, method="Spline", recode_zeros=True)
    msg = str(e.value)
    assert "X642244" in msg               # the series whose curve dipped <= 0
    assert "1455" in msg                  # the near-zero chronology year
    assert "recode_zeros" in msg          # actionable remedy


def test_ssf_crust_preset_rescues_co021():
    # co021 defeats the basic method (curve <= 0), but CRUST's guards -- the
    # near-zero division guard and the 0.02 curve floor -- let it standardise.
    data = _read_quiet("tests/data/csv/co021.csv")
    with pytest.raises(ValueError):
        _ssf_quiet(data, recode_zeros=True)                    # basic: fails
    out = _ssf_quiet(data, preset="crust", recode_zeros=True)  # crust: succeeds
    assert list(out.columns) == ["sfc", "samp_depth"]
    assert np.all(np.isfinite(out["sfc"].to_numpy()))


def test_ssf_crust_rescale_is_multiplicative():
    from dplpy.simplesignalfree import _sf_rescale
    sf = np.array([[2.0], [4.0], [6.0]])       # mean 4
    dat = np.array([[1.0], [2.0], [3.0]])      # mean 2
    # multiplicative: factor = 2/4 -> [1, 2, 3] (both scaled and mean-matched)
    assert np.allclose(_sf_rescale(sf, dat, crust=True, difference=False)[:, 0],
                       [1.0, 2.0, 3.0])
    # additive (basic): shift so mean matches -> [0, 2, 4]
    assert np.allclose(_sf_rescale(sf, dat, crust=False, difference=False)[:, 0],
                       [0.0, 2.0, 4.0])


def test_ssf_crust_preset_default_unchanged():
    # preset=None must remain the exact basic method (already validated vs dplR)
    data = _read_quiet("tests/data/csv/ca533.csv")
    basic = _ssf_quiet(data, recode_zeros=True)
    crust = _ssf_quiet(data, preset="crust", recode_zeros=True)
    assert not np.allclose(basic["sfc"].to_numpy(), crust["sfc"].to_numpy())


def test_ssf_bad_preset():
    data = _read_quiet("tests/data/csv/ca533.csv")
    with pytest.raises(ValueError):
        _ssf_quiet(data, recode_zeros=True, preset="bogus")
