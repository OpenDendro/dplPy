"""Regression tests for the 2026-09 fidelity-audit fixes.

Each test pins one audit finding so the specific defect cannot silently return.
See dev/fidelity_audit_findings_2026-09-09.md for the findings these guard.
"""
import inspect
from math import floor

import pandas as pd
import pytest

import dplpy as dpl
from dplpy.smoothingspline import get_period
from dplpy.simplesignalfree import ssf
from dplpy.xdate_report import _format_part5
from dplpy.xdate import _cofecha_crit


# --- A2 #7: get_period wavelength conventions --------------------------------
def test_get_period_conventions():
    # None -> the default 67% spline, floored (matches dplR detrend.series).
    assert get_period(None, 100) == floor(0.67 * 100) == 67
    assert get_period(None, 133) == floor(0.67 * 133)
    # period < 0 -> ARSTAN "percent spline": |value|/100 * n (idt/100*n).
    assert get_period(-10, 100) == pytest.approx(10.0)
    assert get_period(-50, 200) == pytest.approx(100.0)
    # 0 < period <= 1 -> fraction of length; period > 1 -> fixed wavelength.
    assert get_period(0.1, 100) == pytest.approx(10.0)
    assert get_period(40, 100) == 40


# --- A4 #16: chron_stabilized default overlap ratio is exactly 1/3 -----------
def test_chron_stabilized_default_min_seg_ratio_is_one_third():
    default = inspect.signature(dpl.chron_stabilized).parameters["min_seg_ratio"].default
    # dplR drops overlaps below win_length/3, so the default must be exactly 1/3,
    # not the rounded 0.33 that shifted the cutoff by a year at some windows.
    assert default == pytest.approx(1 / 3, abs=1e-12)
    assert default != 0.33


# --- A3 #10: PART 5 critical value is derived from the window, not hardcoded --
def _minimal_rep(slide_period, preset="COFECHA"):
    return {
        "seg_corr": pd.DataFrame(),
        "flags": {},
        "bins": [],
        "slide_period": slide_period,
        "avg_seg_corr": None,
        "preset": preset,
    }


def test_xdate_report_part5_crit_scales_with_window():
    # Standard 50-year window -> COFECHA's .3281.
    out50 = _format_part5(_minimal_rep(50))
    assert ".3281" in out50
    assert "50-year window" in out50

    # A different window must print its own critical value, not .3281.
    out40 = _format_part5(_minimal_rep(40))
    expected = ("%.4f" % _cofecha_crit(40)).lstrip("0")
    assert expected in out40
    assert "40-year window" in out40
    assert ".3281" not in out40

    # The dplR-faithful (non-preset) path is unchanged: p = 0.05, no crit literal.
    out_default = _format_part5(_minimal_rep(50, preset=None))
    assert "pcrit = 0.05" in out_default
    assert "0.3281" not in out_default


# --- A2 #6: ssf mad-threshold warning states the correct range ---------------
def test_ssf_mad_warning_states_correct_range():
    consts = [c for c in ssf.__code__.co_consts if isinstance(c, str)]
    # The recommended range must match the code's own check (1e-4 < x < 1e-3) and
    # dplR's ssf() ("0.0001 to 0.001"), not the old contradictory "1e-5 and 1e-4".
    assert any("1e-4 and 1e-3" in c for c in consts)
    assert not any("1e-5 and 1e-4" in c for c in consts)
