"""Tests for dplPy's friendly warning output (dpl.set_friendly_warnings).

dplPy raises its own warnings as ``DplpyWarning`` (a ``UserWarning`` subclass)
and, by default, prints them in a clean ``dplPy: <message>`` form instead of
Python's file/line/source-echo chrome -- while leaving every other library's
warnings untouched.
"""
import warnings

import pytest

import dplpy as dpl
from dplpy._validate import _warn, DplpyWarning


@pytest.fixture(autouse=True)
def _restore_formatter():
    """Each test may toggle the global formatter; restore the exact object that
    was active at setup (the import-time friendly formatter) afterwards so tests
    stay independent and leave the session's warning display as it found it."""
    saved = warnings.formatwarning
    try:
        yield
    finally:
        warnings.formatwarning = saved


def test_category_is_a_userwarning_subclass():
    # existing filters written against UserWarning must keep catching dplPy's
    assert issubclass(DplpyWarning, UserWarning)


def test_warn_defaults_to_dplpy_category():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _warn("hello")
    assert len(rec) == 1
    assert rec[0].category is DplpyWarning


def test_friendly_is_on_by_default():
    # importing dplpy installed the friendly formatter
    assert getattr(warnings.formatwarning, "_dplpy_friendly", False)


def test_friendly_format_for_dplpy_warning():
    out = warnings.formatwarning(
        DplpyWarning("the message"), DplpyWarning, "readers.py", 433, "src line")
    assert out == "dplPy: the message\n"
    # no chrome: no filename, no line number, no echoed source line
    assert "readers.py" not in out and "433" not in out and "src line" not in out


def test_other_libraries_warnings_are_untouched():
    out = warnings.formatwarning(
        UserWarning("numpy says hi"), UserWarning, "numpy/core.py", 12, "x = 1")
    assert not out.startswith("dplPy:")
    assert "numpy/core.py" in out and "12" in out


def test_deprecationwarning_stays_standard():
    out = warnings.formatwarning(
        DeprecationWarning("old= is deprecated"), DeprecationWarning, "f.py", 5, "y")
    assert not out.startswith("dplPy:")
    assert "DeprecationWarning" in out


def test_toggle_off_restores_default_then_on_reinstalls():
    dpl.set_friendly_warnings(False)
    assert not getattr(warnings.formatwarning, "_dplpy_friendly", False)
    off = warnings.formatwarning(
        DplpyWarning("msg"), DplpyWarning, "f.py", 5, "line")
    assert not off.startswith("dplPy:")       # standard chrome is back
    assert "DplpyWarning" in off and "f.py" in off

    dpl.set_friendly_warnings(True)
    assert getattr(warnings.formatwarning, "_dplpy_friendly", False)
    on = warnings.formatwarning(
        DplpyWarning("msg"), DplpyWarning, "f.py", 5, "line")
    assert on == "dplPy: msg\n"


def test_reinstall_is_idempotent_and_does_not_wrap_itself():
    dpl.set_friendly_warnings(True)
    first = warnings.formatwarning
    dpl.set_friendly_warnings(True)           # no-op, must not double-wrap
    assert warnings.formatwarning is first
    # a foreign warning still delegates exactly once to the real default
    out = warnings.formatwarning(
        UserWarning("hi"), UserWarning, "f.py", 1, "line")
    assert out.count("hi") == 1 and not out.startswith("dplPy:")


def test_off_leaves_a_third_party_formatter_alone():
    # if some other code took over the formatter, turning friendly OFF must not
    # clobber their replacement
    dpl.set_friendly_warnings(False)

    def custom(message, category, filename, lineno, line=None):
        return "CUSTOM: %s\n" % message

    warnings.formatwarning = custom
    dpl.set_friendly_warnings(False)          # we are not installed -> no-op
    assert warnings.formatwarning is custom


def test_reactivating_a_saved_friendly_formatter_still_delegates():
    # Regression: disabling once nulled the shared delegate, so bringing a saved
    # friendly formatter back by hand (as a test fixture does, and as happens
    # under pytest's own warning capture) left it delegating to None and crashed
    # on the next non-dplPy warning. The predecessor is now held per-install, so
    # a re-activated friendly formatter must still format a foreign warning.
    saved = warnings.formatwarning            # the friendly formatter
    assert getattr(saved, "_dplpy_friendly", False)
    dpl.set_friendly_warnings(False)          # restores the delegate
    warnings.formatwarning = saved            # re-activate the friendly one by hand
    out = warnings.formatwarning(
        UserWarning("foreign"), UserWarning, "f.py", 1, "line")
    assert "foreign" in out and not out.startswith("dplPy:")


def test_filter_mutes_only_dplpy():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=DplpyWarning)
        _warn("dplPy one -- silenced")
        warnings.warn("foreign one -- kept", UserWarning)
    assert len(rec) == 1
    assert str(rec[0].message) == "foreign one -- kept"
