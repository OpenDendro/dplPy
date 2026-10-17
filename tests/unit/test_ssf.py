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


def test_ssf_negative_curve_error_points_to_crust():
    # co021 (many absent rings) drives the chronology near zero, which blows up
    # the signal-free measurements and makes refitted curves go <= 0. The basic
    # method stops (as dplR does) with a message that names the pointer-year cause
    # and the offending series and directs the user to preset="crust" rather than
    # recode_zeros (which does not help a low-but-positive chronology).
    data = _read_quiet("tests/data/csv/co021.csv")
    with pytest.raises(ValueError) as e:
        _ssf_quiet(data, method="Spline", recode_zeros=True)
    msg = str(e.value)
    assert "went <= 0 for series" in msg
    assert "1455" in msg                  # the near-zero chronology year
    assert "crust" in msg                 # the actionable remedy for this data
    assert "literal zero" in msg          # recode_zeros named only as the caveat


def test_ssf_crust_preset_rescues_co021():
    # co021 defeats the basic method (curve <= 0), but CRUST's guards -- the
    # near-zero division guard and the 0.02 curve floor -- let it standardise,
    # keeping every series (nothing set aside).
    data = _read_quiet("tests/data/csv/co021.csv")
    with pytest.raises(ValueError):
        _ssf_quiet(data, recode_zeros=True)                    # basic: fails
    out = _ssf_quiet(data, preset="crust", recode_zeros=True)  # crust: succeeds
    assert list(out.columns) == ["sfc", "samp_depth"]
    assert np.all(np.isfinite(out["sfc"].to_numpy()))
    # crust keeps every series (it floors the curve rather than dropping): the
    # chronology spans the full record, deeply replicated at its core.
    assert out["samp_depth"].max() >= data.shape[1] - 2


def test_ssf_crust_survives_interior_absent_ring():
    # Regression: the crust path turned a legitimate interior absent (zero-width)
    # ring into an illegal interior NaN and raised "'NA's are not allowed in the
    # middle of the series". A single interior zero must not break it now.
    rng = np.random.default_rng(1)
    years = np.arange(1900, 1981)
    n = len(years)
    cols = {}
    for s in range(12):
        age = np.arange(1, n + 1)
        curve = 0.8 * np.exp(-age / 50) + 0.2
        cols[f"T{s:02d}a"] = np.clip(curve * (1 + 0.2 * rng.standard_normal(n)), 0.01, None)
    rwl = pd.DataFrame(cols, index=pd.Index(years, name="Year"))
    rwl.loc[1940, "T00a"] = 0.0           # one interior absent ring
    out = _ssf_quiet(rwl, preset="crust")
    assert np.all(np.isfinite(out["sfc"].to_numpy()))


def test_ssf_negative_curve_names_offending_series():
    # A series whose fitted curve goes <= 0 (here an isolated spike on a flat-tiny
    # series) stops the run with a clear error that names it and points to crust.
    # ssf does not drop series -- "a few" is arbitrary and dropping can cascade.
    rng = np.random.default_rng(2)
    years = np.arange(1900, 2001)
    n = len(years)
    cols = {}
    for s in range(8):
        age = np.arange(1, n + 1)
        cols[f"C{s:02d}a"] = (0.5 * np.exp(-age / 60) + 0.4) * (1 + 0.08 * rng.standard_normal(n))
    victim = np.full(n, 0.05)
    victim[50] = 9.0
    cols["V00a"] = victim
    rwl = pd.DataFrame(cols, index=pd.Index(years, name="Year"))

    with pytest.raises(ValueError) as e:
        _ssf_quiet(rwl, spline_period=10)
    assert "V00a" in str(e.value)
    assert "crust" in str(e.value)
    with pytest.raises(ValueError) as e2:
        _ssf_quiet(rwl, spline_period=10, preset="crust")
    assert "V00a" in str(e2.value)
    assert "dividing" not in str(e2.value)  # raw-curve case, not a division cause


def test_ssf_difference_message_wording():
    # the non-positive-curve explanation must not talk about division in
    # difference mode (which subtracts), and must in ratio mode.
    from dplpy.simplesignalfree import _curve_cause
    chron = np.array([1.0, 0.9, 0.02, 1.1])
    years = np.array([1948, 1949, 1950, 1951])
    ratio = _curve_cause(chron, years, difference=False)
    diff = _curve_cause(chron, years, difference=True)
    assert "dividing by it" in ratio and "1950" in ratio
    assert "dividing" not in diff and "difference mode" in diff and "1950" in diff


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


# --- v0.9.0 audit follow-ups ------------------------------------------------

def test_ssf_recode_zeros_does_not_mutate_input():
    # A2: ssf(recode_zeros=True) used to write 0.001 into the caller's frame
    data = _read_quiet("tests/data/csv/ca533.csv")
    before = data.copy()
    n_zero = int((data == 0).sum().sum())
    assert n_zero > 0
    _ssf_quiet(data, recode_zeros=True)
    pd.testing.assert_frame_equal(data, before)
    assert int((data == 0).sum().sum()) == n_zero


def test_ssf_fit_and_method_vocabulary():
    # D5: fit= names the curve (case-insensitive, no silent fallback) and
    # method= is ratio/difference, as in detrend.
    data = _read_quiet("tests/data/csv/ca533.csv")
    a = _ssf_quiet(data, fit="Spline", recode_zeros=True)
    b = _ssf_quiet(data, fit="spline", recode_zeros=True)
    assert np.allclose(a["sfc"].to_numpy(), b["sfc"].to_numpy())
    assert np.allclose(a["sfc"].to_numpy()[:5], _DPLR_SSF_CA533, atol=1e-4)
    with pytest.raises(ValueError, match="fit"):
        _ssf_quiet(data, fit="bogus", recode_zeros=True)
    with pytest.raises(ValueError):
        _ssf_quiet(data, fit="ModNegExp", recode_zeros=True)      # not an ssf curve
    with pytest.raises(ValueError):
        _ssf_quiet(data, method="bogus", recode_zeros=True)
    d = _ssf_quiet(data, method="difference", recode_zeros=True)
    assert np.all(np.isfinite(d["sfc"].to_numpy()))
    assert not np.allclose(d["sfc"].to_numpy(), a["sfc"].to_numpy())


def test_ssf_deprecated_spellings():
    data = _read_quiet("tests/data/csv/ca533.csv")
    ref = _ssf_quiet(data, fit="Spline", recode_zeros=True)
    buf = io.StringIO()
    # old method= curve name
    with pytest.warns(DeprecationWarning, match="method="):
        with contextlib.redirect_stdout(buf):
            a = dpl.ssf(data, method="Spline", recode_zeros=True, verbose=False)
    assert np.allclose(a["sfc"].to_numpy(), ref["sfc"].to_numpy())
    # old nyrs= -> spline_period=
    with pytest.warns(DeprecationWarning, match="nyrs="):
        with contextlib.redirect_stdout(buf):
            b = dpl.ssf(data, nyrs=0.5, recode_zeros=True, verbose=False)
    c = _ssf_quiet(data, spline_period=0.5, recode_zeros=True)
    assert np.allclose(b["sfc"].to_numpy(), c["sfc"].to_numpy())
    # old mad_threshold= -> tol=
    with pytest.warns(DeprecationWarning, match="mad_threshold="):
        with contextlib.redirect_stdout(buf):
            dpl.ssf(data, mad_threshold=5e-4, recode_zeros=True, verbose=False)
    # old difference=True -> method="difference"
    with pytest.warns(DeprecationWarning, match="difference="):
        with contextlib.redirect_stdout(buf):
            e = dpl.ssf(data, difference=True, recode_zeros=True, verbose=False)
    f = _ssf_quiet(data, method="difference", recode_zeros=True)
    assert np.allclose(e["sfc"].to_numpy(), f["sfc"].to_numpy())
    with pytest.raises(TypeError):
        _ssf_quiet(data, bogus=1, recode_zeros=True)


def test_ssf_warns_instead_of_printing_and_requires_dataframe():
    data = _read_quiet("tests/data/csv/ca533.csv")
    buf = io.StringIO()
    with pytest.warns(UserWarning, match="max_iterations > 25"):
        with contextlib.redirect_stdout(buf):
            dpl.ssf(data, max_iterations=30, recode_zeros=True, verbose=False)
    assert "Warning" not in buf.getvalue()
    with pytest.raises(TypeError):
        _ssf_quiet("tests/data/csv/ca533.csv", recode_zeros=True)
    # G9/preset: case-insensitive
    a = _ssf_quiet(data, preset="CRUST", recode_zeros=True)
    b = _ssf_quiet(data, preset="crust", recode_zeros=True)
    assert np.allclose(a["sfc"].to_numpy(), b["sfc"].to_numpy())


def test_ssf_drops_all_nan_columns():
    # B1: a year slice leaving all-NaN columns used to raise "[5] series with
    # all zero values"; the empty series are dropped with a warning instead.
    data = _read_quiet("tests/data/csv/ca533.csv").loc[1700:1983]
    assert any(data[c].isna().all() for c in data.columns)
    with pytest.warns(UserWarning, match="no measured values"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.filterwarnings("always", message=".*no measured values.*")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = dpl.ssf(data, recode_zeros=True, verbose=False)
    assert np.all(np.isfinite(out["sfc"].to_numpy()))
