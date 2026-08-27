import numpy as np
import pytest
import dplpy as dpl

# ar_func (AR prewhitening residuals) and autoreg (fitted AR coefficients) on a
# real series. Values are pinned against the current dplR-validated output.


def _cam011():
    return dpl.readers("./tests/data/csv/ca533.csv")["CAM011"].dropna()


def test_ar_func_prewhitens():
    one = _cam011()
    arp = dpl.ar_func(one)
    # ar_func returns the AR residuals with the first `order` (here 5) rings
    # dropped, so it is shorter than the input.
    assert len(arp) == len(one) - 5
    assert np.all(np.isfinite(arp.to_numpy()))
    assert arp.dropna().iloc[-1] == pytest.approx(0.657523, abs=1e-5)


def test_autoreg_coefficients():
    one = _cam011()
    coefs = dpl.autoreg(one)
    assert len(coefs) == 6                                 # selected AR order + intercept
    assert np.all(np.isfinite(coefs.to_numpy()))
    assert float(coefs.iloc[0]) == pytest.approx(0.058327, abs=1e-5)
    assert float(np.sum(coefs)) == pytest.approx(0.921575, abs=1e-5)
