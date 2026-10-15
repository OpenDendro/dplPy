import numpy as np
import pytest
import dplpy as dpl

# ar_func (AR prewhitening residuals) and autoreg (fitted AR coefficients) on a
# real series. The OLS values are pinned against the earlier dplR-validated
# output (R's ar(method="ols"), global AIC minimum, order.max=5); the default
# path is now Yule-Walker, ar_max=10, first local AIC minimum (ARSTAN's rule),
# pinned as a regression guard.


def _cam011():
    return dpl.readers("./tests/data/csv/ca533.csv")["CAM011"].dropna()


def test_ar_func_prewhitens_ols():
    one = _cam011()
    arp = dpl.ar_func(one, method="ols", ar_max=5, first_aic_min=False)
    # ar_func returns the AR residuals with the first `order` (here 5) rings
    # dropped, so it is shorter than the input.
    assert len(arp) == len(one) - 5
    assert np.all(np.isfinite(arp.to_numpy()))
    assert arp.dropna().iloc[-1] == pytest.approx(0.657523, abs=1e-5)


def test_autoreg_coefficients_ols():
    one = _cam011()
    coefs = dpl.autoreg(one, method="ols", ar_max=5, first_aic_min=False)
    assert len(coefs) == 6                                 # selected AR order + intercept
    assert np.all(np.isfinite(coefs.to_numpy()))
    assert float(coefs.iloc[0]) == pytest.approx(0.058327, abs=1e-5)
    assert float(np.sum(coefs)) == pytest.approx(0.921575, abs=1e-5)


def test_ar_func_default_is_yule_walker_first_aic_min():
    one = _cam011()
    arp = dpl.ar_func(one)
    # YW with the first-local-AIC-minimum rule selects order 6 for CAM011
    assert len(arp) == len(one) - 6
    assert np.all(np.isfinite(arp.to_numpy()))
    assert arp.iloc[-1] == pytest.approx(0.660797, abs=1e-5)
    coefs = np.asarray(dpl.autoreg(one), dtype=float)
    assert len(coefs) == 7
    assert coefs[0] == pytest.approx(0.053849, abs=1e-5)
    assert float(np.sum(coefs)) == pytest.approx(0.931348, abs=1e-5)
