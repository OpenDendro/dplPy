import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

import dplpy as dpl


# A tiny hand-verifiable dataset (5 years x 4 series).
#   A rises every year;  B and C are anti-phased;  D has flat (no-change) years.
def _toy():
    return pd.DataFrame(
        {"A": [1.0, 2, 3, 4, 5],
         "B": [1.0, 2, 1, 3, 2],
         "C": [3.0, 2, 3, 1, 2],
         "D": [2.0, 2, 3, 3, 5]},
        index=pd.Index([1, 2, 3, 4, 5], name="Year"),
    )


def _glk(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # silence the <50 overlap warning
        return dpl.glk(data, **kw)


def _sgc(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dpl.sgc(data, **kw)


# --- hand-computed values -----------------------------------------------------
def test_glk_hand_values():
    g = _glk(_toy(), overlap=3)["glk_mat"]
    # signs: A=[+,+,+,+] B=[+,-,+,-] C=[-,+,-,+] D=[0,+,0,+]
    assert g.loc["A", "B"] == pytest.approx(0.5)
    assert g.loc["A", "C"] == pytest.approx(0.5)
    assert g.loc["A", "D"] == pytest.approx(0.75)
    assert g.loc["B", "C"] == pytest.approx(0.0)     # perfectly anti-phased
    assert g.loc["B", "D"] == pytest.approx(0.25)
    assert g.loc["C", "D"] == pytest.approx(0.75)
    assert g.loc["A", "A"] == pytest.approx(1.0)     # diagonal forced to 1


def test_sgc_hand_values():
    r = _sgc(_toy(), overlap=3)
    s, ss = r["sgc_mat"], r["ssgc_mat"]
    # A vs D: 2 synchronous, 2 semi-synchronous out of 4 -> 0.5 / 0.5
    assert s.loc["A", "D"] == pytest.approx(0.5)
    assert ss.loc["A", "D"] == pytest.approx(0.5)
    # A vs B: 2 synchronous, 0 semi -> 0.5 / 0.0
    assert s.loc["A", "B"] == pytest.approx(0.5)
    assert ss.loc["A", "B"] == pytest.approx(0.0)
    # B vs C: fully opposite -> 0 / 0
    assert s.loc["B", "C"] == pytest.approx(0.0)
    assert ss.loc["B", "C"] == pytest.approx(0.0)


def test_overlap_counts():
    r = _glk(_toy(), overlap=3)
    # every pair shares all 4 growth-change intervals
    off = r["overlap"].to_numpy()
    assert np.all(off == 4)


# --- the identity that ties glk and sgc together ------------------------------
def test_glk_equals_sgc_plus_half_ssgc():
    data = dpl.readers("./tests/data/csv/ca533.csv")
    g = _glk(data, overlap=3)["glk_mat"].to_numpy()
    r = _sgc(data, overlap=3)
    combo = r["sgc_mat"].to_numpy() + r["ssgc_mat"].to_numpy() / 2.0
    m = ~np.isnan(g)
    assert np.allclose(g[m], combo[m], rtol=1e-12, atol=1e-12)


# --- structural properties ----------------------------------------------------
def test_matrices_are_symmetric():
    r = _glk(_toy(), overlap=3)
    assert np.allclose(r["glk_mat"].to_numpy(), r["glk_mat"].to_numpy().T)
    assert np.allclose(r["overlap"].to_numpy(), r["overlap"].to_numpy().T)
    s = _sgc(_toy(), overlap=3)
    assert np.allclose(s["sgc_mat"].to_numpy(), s["sgc_mat"].to_numpy().T)
    assert np.allclose(s["ssgc_mat"].to_numpy(), s["ssgc_mat"].to_numpy().T)


def test_sgc_diagonal_is_one():
    s = _sgc(_toy(), overlap=3)
    assert np.allclose(np.diag(s["sgc_mat"].to_numpy()), 1.0)
    assert np.allclose(np.diag(s["ssgc_mat"].to_numpy()), 0.0)


def test_keys_and_prob_toggle():
    assert set(_glk(_toy(), overlap=3).keys()) == {"glk_mat", "overlap", "p_mat"}
    assert set(_glk(_toy(), overlap=3, prob=False).keys()) == {"glk_mat", "overlap"}
    assert set(_sgc(_toy(), overlap=3).keys()) == {"sgc_mat", "ssgc_mat", "overlap", "p_mat"}
    assert set(_sgc(_toy(), overlap=3, prob=False).keys()) == {"sgc_mat", "ssgc_mat", "overlap"}


# --- p-values (dplR formula, incl. the p>1 quirk below 0.5) --------------------
def test_p_values_match_formula():
    r = _glk(_toy(), overlap=3)
    g, o, p = r["glk_mat"], r["overlap"], r["p_mat"]
    for a in ["A", "B", "C", "D"]:
        for b in ["A", "B", "C", "D"]:
            s = 1.0 / (2.0 * np.sqrt(o.loc[a, b]))
            expected = 2.0 * (1.0 - ndtr((g.loc[a, b] - 0.5) / s))
            assert p.loc[a, b] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    # glk below 0.5 legitimately yields p > 1 in dplR
    assert p.loc["B", "C"] > 1.0


# --- argument validation (matches dplR) ---------------------------------------
def test_overlap_must_be_integer_ge_3():
    for bad in (2, 3.5, 0, -1):
        with pytest.raises(ValueError):
            dpl.glk(_toy(), overlap=bad)
    with pytest.raises(ValueError):
        dpl.sgc(_toy(), overlap=[3, 4])


def test_low_overlap_warns():
    with pytest.warns(UserWarning):
        dpl.glk(_toy(), overlap=10)


def test_prob_must_be_bool():
    with pytest.raises(ValueError):
        _glk(_toy(), overlap=3, prob="yes")


# --- independent validation against dplR's own glk.legacy (contiguous method) --
def _ref_glk_legacy(df):
    """Transcription of dplR's glk.legacy (common-interval implementation)."""
    X = df.to_numpy(dtype=float)
    n = X.shape[1]
    G = np.full((n, n), np.nan)
    for i in range(n - 1):
        nn1 = np.where(~np.isnan(X[:, i]))[0]
        if len(nn1) < 3:
            continue
        for k in range(i + 1, n):
            nn2 = np.where(~np.isnan(X[:, k]))[0]
            both = np.intersect1d(nn1, nn2)
            m = len(both)
            if m >= 3 and both[-1] - both[0] + 1 == m:      # contiguous only
                d1 = np.sign(np.diff(X[both, i]))
                d2 = np.sign(np.diff(X[both, k]))
                G[i, k] = 1.0 - np.sum(np.abs(d1 - d2)) / (2 * m - 2)
    return G


def test_matches_dplR_glk_legacy_on_ca533():
    # ca533 series are internally contiguous, so the fast glk and dplR's
    # common-interval glk.legacy must agree to machine precision.
    data = dpl.readers("./tests/data/csv/ca533.csv")
    fast = _glk(data, overlap=3)["glk_mat"].to_numpy()
    ref = _ref_glk_legacy(data)
    iu = np.triu_indices_from(ref, k=1)
    f, r = fast[iu], ref[iu]
    m = ~np.isnan(r)
    assert m.sum() > 100                                  # actually compared many pairs
    assert np.allclose(f[m], r[m], rtol=1e-12, atol=1e-12)
