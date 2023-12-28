import dplpy as dpl
import pandas as pd
import pytest
import io

open_wrapper = io.TextIOWrapper(
    io.BytesIO(),
    encoding='cp1252',
    line_buffering=True,
)
open_wrapper.mode = "w"

def test_write_invalid_type_data():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8], 
                                  "Year": [1, 2, 3, 4]})
    
    with pytest.raises(TypeError) as errorMsg:
        dpl.writers(input_df['SeriesA'], "label", "ext")
    expected_msg = "Expected input data to be pandas dataframe, not <class 'pandas.core.series.Series'>"
    assert expected_msg == str(errorMsg.value)


def test_write_invalid_type_label():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8], 
                                  "Year": [1, 2, 3, 4]})
    
    with pytest.raises(TypeError) as errorMsg:
        dpl.writers(input_df, 1, "ext")
    expected_msg = "Expected label to be of type str, not <class 'int'>"
    assert expected_msg == str(errorMsg.value)


def test_write_invalid_type_format():
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8], 
                                  "Year": [1, 2, 3, 4]})
    
    with pytest.raises(TypeError) as errorMsg:
        dpl.writers(input_df, "label", 1)
    expected_msg = "Expected format to be of type str, not <class 'int'>"
    assert expected_msg == str(errorMsg.value)


def test_write_csv(tmpdir):
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8]}, 
                                index=pd.Index(data=[1, 2, 3, 4], name="Year"))
    
    file = tmpdir.join('output.csv')

    dpl.writers(input_df, file.strpath[:-4], "csv")

    expected_csv_lines = ['"Year","SeriesA","SeriesB"\n', 
                          '1,0.1,0.2\n', 
                          '2,0.3,0.4\n',
                          '3,0.5,0.6\n',
                          '4,0.7,0.8\n']

    assert expected_csv_lines == file.readlines()

    

def test_write_rwl(tmpdir):
    input_df = pd.DataFrame(data={"SeriesA": [0.1, 0.3, 0.5, 0.7], 
                                  "SeriesB": [0.2, 0.4, 0.6, 0.8]}, 
                                index=pd.Index(data=[1, 2, 3, 4], name="Year"))
    
    file = tmpdir.join('output.rwl')
    
    dpl.writers(input_df, file.strpath[:-4], "rwl")

    expected_rwl_lines = ['SeriesA    1  0100  0300  0500  0700 -9999\n',
                          'SeriesB    1  0200  0400  0600  0800 -9999\n']

    assert expected_rwl_lines == file.readlines()


#TODO: Add tests for crn and txt