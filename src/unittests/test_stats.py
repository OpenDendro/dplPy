import dplpy as dpl
import pandas as pd
from unittest.mock import patch, Mock
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper

# Data being read:
# SeriesA  1   10 30  50  70  90  110 130 150 999
# SeriesB  1   20 40  60  80  100 120 140 160 999

def mock_auto_reg_fit(self):
    return Mock(**{"params":[1.0, 1.0]})

def mock_readers_output(file_name):
    if file_name == "valid_file":
        return pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))


# Need to mock autoreg
# Need to mock output of readers

@patch('stats.readers')
@patch.object(AutoReg, 'fit', new=mock_auto_reg_fit)
def test_stats_with_inp_string(mock_readers: Mock):
    mock_readers.side_effect = mock_readers_output
    
    expected_df = pd.DataFrame(data={"series": ["SeriesA", "SeriesB"],
                                     "first": [1, 1],
                                     "last": [8, 8],
                                     "year": [8, 8],
                                     "mean": [0.8, 0.9],
                                     "median": [0.8, 0.9],
                                     "stdev": [0.49, 0.49],
                                     "skew": [0.0, 0.0],
                                     "gini": [0.328, 0.292],
                                     "ar1": [1.0, 1.0]
                                     },
                                index=[1, 2])
    results = dpl.stats("valid_file")

    mock_readers.assert_called_once_with("valid_file")
    pd.testing.assert_frame_equal(results, expected_df)


@patch('stats.readers')
@patch.object(AutoReg, 'fit', new=mock_auto_reg_fit)
def test_stats_with_inp_df(mock_readers: Mock):
    mock_readers.side_effect = mock_readers_output
    
    expected_df = pd.DataFrame(data={"series": ["SeriesA", "SeriesB"],
                                     "first": [1, 1],
                                     "last": [8, 8],
                                     "year": [8, 8],
                                     "mean": [0.8, 0.9],
                                     "median": [0.8, 0.9],
                                     "stdev": [0.49, 0.49],
                                     "skew": [0.0, 0.0],
                                     "gini": [0.328, 0.292],
                                     "ar1": [1.0, 1.0]
                                     },
                                index=[1, 2])
    
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                    "SeriesB": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]},
                                    index=pd.Index(data=[1, 2, 3, 4, 5, 6, 7, 8], 
                                                    name="Year"))

    results = dpl.stats(input_df)
    mock_readers.assert_not_called()
    pd.testing.assert_frame_equal(results, expected_df)
