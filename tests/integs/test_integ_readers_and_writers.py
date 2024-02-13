import dplpy as dpl
import pandas as pd
import os

def test_read_and_write_csv(tmp_path):
    ca533 = dpl.readers("./tests/data/csv/ca533.csv")

    write_path = os.path.join(tmp_path,"test_write")
    
    dpl.writers(ca533, write_path, "csv")

    ca533_alt = dpl.readers(write_path + ".csv")

    pd.testing.assert_frame_equal(ca533, ca533_alt)


def test_read_and_write_rwl_no_headers(tmp_path):
    viet001 = dpl.readers("./tests/data/rwl/viet001.rwl")

    write_path = os.path.join(tmp_path, "test_write")

    dpl.writers(viet001, write_path, "rwl")

    viet001_alt = dpl.readers(write_path + ".rwl")

    pd.testing.assert_frame_equal(viet001, viet001_alt)


def test_read_and_write_rwl_with_headers(tmp_path):
    th001 = dpl.readers("./tests/data/rwl/th001.rwl", header=True)

    write_path = os.path.join(tmp_path, "test_write")

    dpl.writers(th001, write_path, "rwl")

    th001_alt = dpl.readers(write_path + ".rwl")

    pd.testing.assert_frame_equal(th001, th001_alt)


def test_read_and_write_long_rwl(tmp_path):
    ca667 = dpl.readers("./tests/data/rwl/ca667.rwl", header=True)

    write_path = os.path.join(tmp_path, "test_write")

    dpl.writers(ca667, write_path, "rwl")

    ca667_alt = dpl.readers(write_path + ".rwl")

    pd.testing.assert_frame_equal(ca667, ca667_alt)

def test_read_and_write_weird_rwl(tmp_path):
    wwr = dpl.readers("./tests/data/rwl/wwr.rwl")

    write_path = os.path.join(tmp_path, "test_write")

    dpl.writers(wwr, write_path, "rwl")

    wwr_alt = dpl.readers(write_path + ".rwl")

    pd.testing.assert_frame_equal(wwr, wwr_alt)