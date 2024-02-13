import dplpy.smoothingspline as spline
from math import pi, cos
from unittest.mock import patch, Mock
import numpy as np

def test_get_param():
    amp = 0.5
    freq = 1/60

    assert spline.get_param(0.5, 60) == 1/(((cos(2 * pi * freq) + 2) * (1 - amp)/(12 * amp * (cos(2 * pi * freq) - 1) ** 2))+ 1)

def test_get_period():
    assert spline.get_period(None, 200) == 134
    assert spline.get_period(-20, 200) == 40
    assert spline.get_period(0.5, 200) == 100
    assert spline.get_period(20, 2000) == 20

def mock_csaps_func(x, y, z, smooth=None):
    freq = 1/5
    amp = 0.5
    if smooth == 1/(((cos(2 * pi * freq) + 2) * (1 - amp)/(12 * amp * (cos(2 * pi * freq) - 1) ** 2))+ 1):
        return y * 0.5
    else:
        return y


@patch('dplpy.smoothingspline.csaps')
def test_spline(mock_csaps: Mock):
    mock_csaps.side_effect = mock_csaps_func
    x = np.arange(10)
    y = np.arange(0.1, 1.0)
    assert np.array_equal(spline.spline(x, y, period=0.5), y * 0.5)
