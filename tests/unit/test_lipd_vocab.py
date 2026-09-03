import pytest

from dplpy import lipd_vocab
from dplpy import _lipd_support


def test_core_dendro_terms_present():
    # the controlled terms confirmed against real lipdverse dendro data
    assert lipd_vocab.ARCHIVE_TYPE == "Wood"
    assert lipd_vocab.lipd_variable("ring_width")["variableName"] == "ringWidth"
    assert lipd_vocab.lipd_variable("ring_width")["units"] == "mm"
    assert lipd_vocab.lipd_variable("trsgi")["variableName"] == "trsgi"
    assert lipd_vocab.lipd_variable("trsgi")["units"] == "unitless"
    assert lipd_vocab.lipd_variable("sample_count")["variableName"] == "sampleCount"
    assert lipd_vocab.lipd_variable("eps")["variableName"] == "EPS"
    assert lipd_vocab.lipd_variable("rbar")["variableName"] == "RBAR"


def test_lipd_variable_returns_a_copy():
    a = lipd_vocab.lipd_variable("trsgi")
    a["units"] = "changed"
    a["past"]["what"] = "changed"
    b = lipd_vocab.lipd_variable("trsgi")
    assert b["units"] == "unitless"          # original table untouched
    assert b["past"]["what"] != "changed"    # nested dict copied too


def test_lipd_variable_unknown_key():
    with pytest.raises(KeyError):
        lipd_vocab.lipd_variable("not_a_variable")


def test_past_long_name_has_nine_comma_separated_fields():
    name = lipd_vocab.past_long_name("ring width", "millimeter")
    assert name.count(",") == 8          # nine fields -> eight commas
    parts = [p.strip() for p in name.split(",")]
    assert parts[0] == "ring width"      # What
    assert parts[3] == "millimeter"      # Units
    assert parts[5] == "Tree Ring"       # Data Type
    assert parts[8] == "N"               # Data Format


def test_past_long_name_keeps_empty_slots():
    # a Detail/Method example still yields all nine positions
    name = lipd_vocab.past_long_name(
        "standardized growth index", "dimensionless",
        detail="detrended with a cubic smoothing spline",
        method="biweight robust mean")
    assert len([p for p in name.split(",")]) == 9


def test_lipd_support_helpers():
    assert isinstance(_lipd_support.has_pylipd(), bool)
    # require_pylipd returns the module if present, else raises a helpful error
    if _lipd_support.has_pylipd():
        assert _lipd_support.require_pylipd() is not None
    else:
        with pytest.raises(ImportError):
            _lipd_support.require_pylipd()
