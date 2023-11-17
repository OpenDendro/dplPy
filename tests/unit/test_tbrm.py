from dplpy.tbrm import tbrm
import pytest

def test_tbrm():
    assert tbrm([-72, 2, 2, 2]) == 2