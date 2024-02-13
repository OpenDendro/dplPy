import dplpy as dpl
import pandas as pd
import pytest
import io
from unittest.mock import patch, Mock

'''
    Test that when given an incorrect file extension, program raises 
    an error with expected message.
'''
def test_wrong_file_extension():
    with pytest.raises(ValueError) as errorMsg:
        dpl.readers("filename.txt")

    wrong_ext_msg = """

Unable to read file, please check that you're using a supported type
Accepted file types are .csv and .rwl

Example usages:
>>> import dplpy as dpl
>>> data = dpl.readers('../tests/data/csv/filename.csv')
>>> data = dpl.readers('../tests/data/rwl/filename.rwl'), header=True
"""
    assert wrong_ext_msg == str(errorMsg.value)


'''
    Mocks output of pd.read_csv to return appropriate dataframe only if the
    parameter used is the expected file name.
'''
def mock_read_csv_output(file_path, skiprows=0):
    if file_path == "correct_file.csv":
        return pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8], 
                                  "Year": [1, 2, 3, 4]})
    return None

'''
    Mocks output of builtins.open to return an io.TextIOWrapper object that contains the lines
    that will be read for processing
'''
def mock_open_output(file_path, open_type):
    # Verify that file is opened in read mode and read mode only
    if open_type != "r":
        wrapper = io.TextIOWrapper(
            io.UnsupportedOperation(),
            encoding='cp1252',
            line_buffering=True,
        )

        wrapper.mode = open_type
        return wrapper
    
    output  = io.BytesIO()
    wrapper = io.TextIOWrapper(
        output,
        encoding='cp1252',
        line_buffering=True,
    )

    if file_path == "valid_rwl_correct_format.rwl":
        wrapper.write("SeriesA 1       10    30    50    70   999\n")
        wrapper.write("SeriesB 1      200   400   600   800 -9999\n")
        wrapper.seek(0,0)
    elif file_path == "valid_rwl_with_headers.rwl":
        wrapper.write("Header line 1\n")
        wrapper.write("Header line 2\n")
        wrapper.write("Header line 3\n")
        wrapper.write("SeriesA 1       10    30    50    70   999\n")
        wrapper.write("SeriesB 1      200   400   600   800 -9999\n")
        wrapper.seek(0,0)
    else:
        raise OSError("File not found")

    wrapper.mode = open_type

    return wrapper 

'''
    Given input file.csv, test that readers produces the expected dataframe.
'''
@patch('pandas.read_csv')
def test_correct_csv_format(mock_read_csv: Mock):
    mock_read_csv.side_effect = mock_read_csv_output
    results = dpl.readers("correct_file.csv")
    mock_read_csv.assert_called_once_with("correct_file.csv", skiprows=0)

    expected_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7], 
                                     "SeriesB": [0.2, 0.4, 0.6, 0.8]}, 
                                     index=pd.Index(data=[1, 2, 3, 4], 
                                                    name="Year")
                                    )
    pd.testing.assert_frame_equal(results, expected_df)

'''
    Given input file valid_rwl_correct_format.rwl, test that readers produces
    the expected dataframe.
'''
@patch('builtins.open')
def test_correct_rwl_format(mock_open: Mock):
    mock_open.side_effect = mock_open_output

    results = dpl.readers("valid_rwl_correct_format.rwl")
    mock_open.assert_called_once_with("valid_rwl_correct_format.rwl", "r")

    expected_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                     "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                                     index=pd.Index(data=[1, 2, 3, 4], 
                                                    name="Year"))
    print(results)
    pd.testing.assert_frame_equal(results, expected_df)

'''
    Given input valid_rwl_correct_format.rwl, and skip_lines=1, test that readers
    produces the expected dataframe.
'''
@patch('builtins.open')
def test_correct_rwl_skip_lines(mock_open: Mock):
    mock_open.side_effect = mock_open_output

    results = dpl.readers("valid_rwl_correct_format.rwl", skip_lines=1)
    print(results)
    mock_open.assert_called_once_with("valid_rwl_correct_format.rwl", "r")

    expected_df = pd.DataFrame(data={"SeriesB": [0.2, 0.4, 0.6, 0.8]},
                                     index=pd.Index(data=[1, 2, 3, 4], 
                                                    name="Year"))
    pd.testing.assert_frame_equal(results, expected_df)

'''
    Given input valid_rwl_correct_format.rwl, and header=True, test that readers
    correctly skips header lines to produce the expected dataframe.
'''
@patch('builtins.open')
def test_correct_rwl_with_headers(mock_open: Mock):
    mock_open.side_effect = mock_open_output

    results = dpl.readers("valid_rwl_with_headers.rwl", header=True)
    mock_open.assert_called_once_with("valid_rwl_with_headers.rwl", "r")

    expected_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7],
                                     "SeriesB": [0.2, 0.4, 0.6, 0.8]},
                                     index=pd.Index(data=[1, 2, 3, 4], 
                                                    name="Year"))
    pd.testing.assert_frame_equal(results, expected_df)