import pytest
import numpy as np
import dplpy.curvefit as curvefit
from unittest.mock import patch, Mock

def test_horizontal():
    input_x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    input_y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    res = curvefit.horizontal(input_x, input_y)
    assert np.array_equal(res, [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45])


def mocked_curvefit_linear(function, x, y):
    return (1, 0.1), None


@patch('dplpy.curvefit.curve_fit')
def test_linear(mock_scipy_curve_fit: Mock):
    mock_scipy_curve_fit.side_effect = mocked_curvefit_linear

    input_x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    input_y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    res = curvefit.linear(input_x, input_y)
    mock_scipy_curve_fit.assert_called()
    assert np.array_equal(res, [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])


def test_fitters_accept_true_age_axis():
    # A15: t= gives the true ring-position / cambial-age axis
    t = np.arange(1.0, 61.0)
    y = 2.0 * np.exp(-0.05 * t) + 0.5
    keep = np.r_[0:20, 30:60]
    curve, info = curvefit.mod_neg_exp(t[keep], y[keep], info=True, t=t[keep])
    assert info["method"] == "NegativeExponential"
    assert np.allclose(curve, y[keep], atol=1e-8)
    curve2, info2 = curvefit.neg_exp_arstan(t[keep], y[keep], info=True, t=t[keep])
    assert info2["method"] == "NegativeExponential"
    assert np.allclose(curve2, y[keep], atol=1e-4)
    with pytest.raises(ValueError):
        curvefit.mod_neg_exp(t, y, t=t[:10])


def test_line_or_mean_fallback_shared():
    # E21: rising series -> line rejected (pos_slope False) -> mean, with warning
    x = np.arange(1.0, 21.0)
    y = x * 0.1 + 1.0
    with pytest.warns(UserWarning, match="series mean"):
        c, meta = curvefit._line_or_mean(x, y, False, "s", "Test", True)
    assert meta["method"] == "Mean" and np.allclose(c, y.mean())
    with pytest.warns(UserWarning, match="linear fit"):
        c, meta = curvefit._line_or_mean(x, y, True, "s", "Test", True)
    assert meta["method"] == "Line"
