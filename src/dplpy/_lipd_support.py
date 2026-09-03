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

# Title: _lipd_support.py
# Description: Optional-dependency plumbing for LiPD import/export. LiPD support
#   is built on `pylipd`, which is NOT a core dplPy dependency (most users don't
#   export LiPD). It is an optional extra: `pip install "dplpy[lipd]"`. The LiPD
#   functions (planned) call require_pylipd() so a missing install fails with a
#   clear, actionable message instead of a bare ImportError.

import importlib.util

_INSTALL_HINT = (
    "LiPD support needs the optional 'pylipd' dependency, which is not installed.\n"
    "Install it with:\n"
    "    pip install \"dplpy[lipd]\"\n"
    "If pylipd's build of 'bibtexparser' fails, install a wheel first:\n"
    "    pip install bibtexparser --only-binary :all:\n"
    "    pip install pylipd"
)


def has_pylipd() -> bool:
    """True if pylipd is importable, without importing it."""
    return importlib.util.find_spec("pylipd") is not None


def require_pylipd():
    """Import and return the pylipd module, or raise ImportError with install
    instructions if it is not available. LiPD functions call this first."""
    try:
        import pylipd
    except ImportError as exc:                       # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc
    return pylipd
