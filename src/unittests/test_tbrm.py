from src.tbrm import tbrm
import pandas as pd
import pytest
import io
from unittest.mock import patch, Mock

def test_tbrm():
    assert tbrm([-72, 2, 2, 2]) == 2