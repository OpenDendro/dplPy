__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2024  OpenDendro

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__license__ = "GNU GPLv3"

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Date: 11/1/2022
# Author: Ifeoluwa Ale
# Title: tbrm.py
# Description: This file contains helper functions which find tukey's biweight robust mean of
#              an array like object.

import numpy as np

def tbrm(data, c=9):
    data = np.asarray(data, dtype=float)
    e = 1 * pow(10, -6)  # regularization epsilon; matches dplR's tbrm C code (C*MAD + 1e-6)
    m = np.median(data)

    s = np.median(np.abs(data - m))

    u = (data - m) / ((c * s) + e)

    w = np.where(np.abs(u) <= 1, (1 - u ** 2) ** 2, 0.0)

    return np.sum(w*data)/np.sum(w)
