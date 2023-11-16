import numpy as np
import dplpy.curvefit as curvefit
from unittest.mock import patch, Mock

def mocked_curvefit_hugershoff(function, x, y, bounds=(), p0=[]):
    return (0.5, 0, 0, 0.2), None

@patch('dplpy.curvefit.curve_fit')
def test_hugershoff(mock_scipy_curve_fit: Mock):
    mock_scipy_curve_fit.side_effect = mocked_curvefit_hugershoff

    input_x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    input_y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    res = curvefit.hugershoff(input_x, input_y)
    mock_scipy_curve_fit.assert_called()
    assert np.array_equal(res, [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])


def mocked_curvefit_negex(function, x, y, bounds=()):
    return (0.5, 0, 0.1), None


@patch('dplpy.curvefit.curve_fit')
def test_negex(mock_scipy_curve_fit: Mock):
    mock_scipy_curve_fit.side_effect = mocked_curvefit_negex

    input_x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    input_y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    res = curvefit.negex(input_x, input_y)
    mock_scipy_curve_fit.assert_called()
    assert np.array_equal(res, [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])


def test_horizontal():
    input_x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    input_y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    res = curvefit.horizontal(input_x, input_y)
    assert np.array_equal(res, [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45])
    

def mocked_curvefit_linear(function, x, y, bounds=[]):
    if np.array_equal(bounds, []):
        return (1, 0.1), None
    elif np.array_equal(bounds, ([-np.inf, -np.inf], [0, np.inf])):
        return (2, 0.2), None
    else:
        return (0, 0), None


@patch('dplpy.curvefit.curve_fit')
def test_linear_bounds_false(mock_scipy_curve_fit: Mock):
    mock_scipy_curve_fit.side_effect = mocked_curvefit_linear

    input_x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    input_y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    res = curvefit.linear(input_x, input_y)
    mock_scipy_curve_fit.assert_called()
    assert np.array_equal(res, [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])


@patch('dplpy.curvefit.curve_fit')
def test_linear_bounds_true(mock_scipy_curve_fit: Mock):
    mock_scipy_curve_fit.side_effect = mocked_curvefit_linear

    input_x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    input_y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    res = curvefit.linear(input_x, input_y, bounds=True)
    mock_scipy_curve_fit.assert_called()
    assert np.array_equal(res, [2.2, 4.2, 6.2, 8.2, 10.2, 12.2, 14.2, 16.2])
