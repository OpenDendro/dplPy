from unittest.mock import patch, Mock
import dplpy as dpl
import pytest
import pandas as pd

def mock_readers_output(file_name):
    if file_name == "valid_file":
        return pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))

def mock_dataframe_summary(self):
    return pd.DataFrame(data={"Col":[1, 2, 3, 4, 5, 6, 7, 8]},
                        index=pd.Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"]))

import importlib
_m_summary = importlib.import_module("dplpy.summary")
_m_readers = importlib.import_module("dplpy.readers")

@patch.object(_m_readers, 'readers')
@patch.object(pd.DataFrame, 'describe', new=mock_dataframe_summary)
def test_summary_given_filename(mock_readers: Mock):
    mock_readers.side_effect = mock_readers_output
    results = dpl.summary("valid_file")

    expected_df = pd.DataFrame(data={"Col":[1, 2, 3, 4, 5, 6, 7, 8]},
                               index=pd.Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"]))

    mock_readers.assert_called_once_with("valid_file")
    pd.testing.assert_frame_equal(results, expected_df)

@patch.object(_m_readers, 'readers')
@patch.object(pd.DataFrame, 'describe', new=mock_dataframe_summary)
def test_summary_given_dataframe(mock_readers: Mock):
    mock_readers.side_effect = mock_readers_output

    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8],
                                                    name="Year"))
    results = dpl.summary(input_df)

    expected_df = pd.DataFrame(data={"Col":[1, 2, 3, 4, 5, 6, 7, 8]},
                               index=pd.Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"]))
    mock_readers.assert_not_called()
    pd.testing.assert_frame_equal(results, expected_df)

def test_summary_given_wrong_type():
    with pytest.raises(TypeError) as errorMsg:
        dpl.summary(1)

    expected_err_msg = ("Input must be a pandas DataFrame or a path (str) to a "
                        "CSV/RWL file, not <class 'int'>.")
    assert expected_err_msg == str(errorMsg.value)

def test_report_returns_stats_table(capsys):
    # D9: report() prints the text report AND returns the per-series stats frame
    # it was built from, so the result is data rather than None.
    import numpy as np
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"S1": rng.uniform(0.2, 2.0, 60), "S2": rng.uniform(0.2, 2.0, 60),
                       "S3": rng.uniform(0.2, 2.0, 60)},
                      index=pd.Index(range(1900, 1960), name="Year"))
    out = dpl.report(df)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3 and "ar1" in out.columns          # one row per series
    assert "Number of dated series: 3" in capsys.readouterr().out
