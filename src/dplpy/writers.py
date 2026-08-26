from __future__ import print_function

__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2021  OpenDendro

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

# Date: 11/17/2021 
# Author: Tyson Lee Swetnam
# Project: OpenDendro- Writers
# Description: Writers for all supported file types (*.CSV, *.RWL, and *.TXT)
# 
# example usages: 
# >>> import dplpy as dpl 
# >>> dpl.writers("./data/in_file.csv", "./data/out_file.rwl")
# >>> dpl.writers("./data/in_file.rwl", "./data/out_file.txt")
# >>> dpl.writers("./data/in_file.txt", "./data/out_file.csv")
# 
# example command line application:
# $ python src/dplpy.py writer --input ./data/in_file.csv --output ./data/out_file.rwl
#
# module uses two arguments: input file and output file 
# define `writer` module as a definition function
# input is expected to be a file path with file name and extension

import pandas as pd
import numpy as np
from .chron import chron
from .detrend import detrend

def writers(data: pd.DataFrame, label: str, format: str, header=None,
            chronology_type="standard", column="std", prec=0.001, gaps=-99):
    """ Output dplpy datasets to .csv, .rwl and .crn files.

    Extended Summary
    ----------------
    Given a pandas dataframe, this function writes its contents to a .csv, .rwl
    or .crn file as indicated by the `format` parameter. The file will be created
    in the same directory unless a different path is included in `label`.

    For 'csv' and 'rwl', `data` is a ring-width dataframe (years x series). For
    'crn', `data` is a *chronology* (the output of dpl.chron() -- a dataframe with
    an index column such as 'std'/'res' and a 'samp_depth' column), and a
    `header` of site metadata is REQUIRED to populate the Tucson .crn header.

    Parameters
    ----------
    data : pandas dataframe
        ring widths (csv/rwl) or a chronology from dpl.chron() (crn).
    label : str
        name (can include file path) to give the file; no extension.
    format : str
        'csv', 'rwl', 'crn', or 'txt'.
    header : dict, required for format='crn'
        site metadata with the keys: 'site_id', 'site_name', 'species_code',
        'state_country', 'species', 'elevation', 'latitude', 'longitude',
        'investigators'; optionally 'completion_date'. First/last year are taken
        from the chronology.
    chronology_type : str, default 'standard'
        for format='crn', the ITRDB chronology type written in header record 2:
        'standard' (blank code), 'arstan' ('A'), or 'residual' ('R').
    column : str, default 'std'
        for format='crn', which chronology column to write (e.g. 'std', 'res').
    prec : float, default 0.001
        for format='rwl', the measurement precision in mm: 0.001 (values written
        x1000, end-of-series marker -9999) or 0.01 (values x100, end marker 999).
        Mirrors dplR's write.tucson.
    gaps : {int, "split"}, default -99
        for format='rwl', how to encode a *true interior gap* (a NaN inside a
        series -- missing measurement, as distinct from a real 0, which is a ring
        that was locally absent that year and is always written as 0):
        a negative integer (Ed Cook's ARSTAN convention, default -99; also e.g.
        -9) writes that sentinel in the gap within a continuous block -- it must
        be negative so it is not read as a ring width, and must not be a stop
        marker; dplPy's reader turns any such negative back into NaN on read.
        "split" instead closes the block with the end marker and reopens a new
        block at the next present year (no sentinel in the file; gap still reads
        back as NaN).

    Returns
    -------
    None

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected input data to be pandas dataframe, not " + str(type(data)))

    if not isinstance(label, str):
        raise TypeError("Expected label to be of type str, not " + str(type(label)))

    if not isinstance(format, str):
        raise TypeError("Expected format to be of type str, not " + str(type(format)))

    filename = label + "." + format
    print("Writing to " + filename)
    output = open(filename, "w")
    try:
        if format == "csv":
            write_csv(data, output)
        elif format == "rwl":
            write_rwl(data, output, prec=prec, gaps=gaps)
        elif format == "crn":
            write_crn(data, output, header=header,
                      chronology_type=chronology_type, column=column)
        elif format == "txt":
            write_txt(data, output)
        else:
            raise ValueError("Invalid file format given as parameter. Accepted file formats are csv, rwl, crn and txt")
    finally:
        output.close()
    print("Done.")


def conv_data(data):
    if np.isnan(data):
        return "NA"
    else:
        return str(data)


def write_csv(data, file):
    file.write('"Year","')
    file.write('","'.join(data.columns.tolist()))
    file.write('"\n')

    for year, row in data.iterrows():
        file.write(str(year))
        file.write(",")
        file.write(",".join(map(conv_data, row)))
        file.write('\n')


def write_rwl(data, file, prec=0.001, gaps=-99):
    """Write a ring-width dataframe (years x series -- raw widths OR detrended
    RWI series, whichever you have) to a Tucson decadal .rwl file.

    ``prec`` is the measurement precision in mm and mirrors dplR's write.tucson:
    0.001 writes values * 1000 with a -9999 end-of-series marker; 0.01 writes
    values * 100 with a 999 end marker. Values are right-justified in 6-column
    fields (the standard space-padded encoding).

    A real 0 in ``data`` is a *locally absent ring* (a ring that did not form that
    year) and is always written as 0. ``gaps`` controls how a *true interior gap*
    -- a NaN inside a series' span, i.e. a missing measurement -- is encoded:

    * a negative integer (Ed Cook's ARSTAN convention, default ``-99``; also e.g.
      ``-9``): write that sentinel in the gap within one continuous block. It must
      be negative so the reader does not mistake it for a ring width, and must not
      collide with a stop marker; dplPy's reader turns any such negative back into
      NaN on read.
    * ``"split"``: close the current block with the end marker and reopen a new
      block at the next present year. Nothing is written in the gap, and reading
      the file back restores the gap as NaN (dplPy's convention).
    """
    if prec == 0.01:
        rproc, stop_code = 100, 999
    elif prec == 0.001:
        rproc, stop_code = 1000, -9999
    else:
        raise ValueError("prec must be 0.01 or 0.001, got " + repr(prec))

    if gaps != "split":
        # A gap sentinel: must be a plain integer, negative (so it is never read
        # as a width), and not a value the reader treats specially (999 / -9999
        # are stop / precision-shift markers at these precisions).
        if isinstance(gaps, bool) or not isinstance(gaps, (int, np.integer)):
            raise ValueError('gaps must be "split" or a negative integer '
                             '(e.g. -9 or -99), got ' + repr(gaps))
        gaps = int(gaps)
        if gaps >= 0:
            raise ValueError("gaps sentinel must be negative so it is not read as "
                             "a ring width, got " + repr(gaps))
        if gaps in (-9999, 999, stop_code):
            raise ValueError("gaps sentinel " + repr(gaps) + " collides with a "
                             "stop marker; use e.g. -9 or -99")

    def head(name, year):
        # The series id + year end at column 12. A BC (negative) year needs one
        # extra column, taken from the id field.
        if year < 0:
            return str(name).ljust(7) + str(year).rjust(5)
        return str(name).ljust(8) + str(year).rjust(4)

    def cell(value):
        return "%6d" % int(value)

    for series in data.columns:
        col = data[series]
        start = col.first_valid_index()
        end = col.last_valid_index()
        if start is None:                           # all-NaN series: skip
            continue
        i, end = int(start), int(end)
        file.write(head(series, i))
        while i <= end:
            if np.isnan(col[i]) and gaps == "split":
                # split mode: close this block, skip the NaN run, reopen at the
                # next present year.
                file.write(cell(stop_code) + "\n")
                while i <= end and np.isnan(col[i]):
                    i += 1
                if i <= end:
                    file.write(head(series, i))
                continue
            if np.isnan(col[i]):
                file.write(cell(gaps))              # sentinel; stays in the block
            else:
                file.write(cell(round(col[i] * rproc)))
            i += 1
            if i % 10 == 0:                         # new decade line
                file.write("\n" + head(series, i))
        file.write(cell(stop_code) + "\n")


# Required site-metadata keys for the .crn header, and the ITRDB chronology
# type codes written in header record 2 (columns 62-63).
_CRN_HEADER_KEYS = ["site_id", "site_name", "species_code", "state_country",
                    "species", "elevation", "latitude", "longitude",
                    "investigators"]
_CRN_TYPE_CODE = {"standard": "  ", "arstan": "A ", "residual": "R "}


def write_crn(chron_data, file, header=None, chronology_type="standard", column="std"):
    """Write a chronology to a Tucson .crn file.

    ``chron_data`` is a chronology -- the output of dpl.chron() -- with an index
    column (``column``, default 'std') and a 'samp_depth' column, indexed by
    year. Index values are written as integers = round(index, 3) * 1000 (so 1000
    is mean growth), with the sample depth alongside, in the ITRDB decadal layout;
    the missing-value code is 9990. ``header`` (a dict of site metadata) is
    required, and populates the three ITRDB header records.
    """
    if column not in chron_data.columns:
        raise ValueError("chronology column '" + str(column) + "' not found; "
                         "available columns: " + str(list(chron_data.columns)))
    if "samp_depth" not in chron_data.columns:
        raise ValueError("chronology must have a 'samp_depth' column -- pass the "
                         "output of dpl.chron().")
    if not isinstance(header, dict):
        raise ValueError("writing a .crn file requires a 'header' dict of site "
                         "metadata with keys: " + ", ".join(_CRN_HEADER_KEYS))
    missing = [k for k in _CRN_HEADER_KEYS if k not in header]
    if missing:
        raise ValueError("header is missing required key(s): " + ", ".join(missing))
    ct = str(chronology_type).lower()
    if ct not in _CRN_TYPE_CODE:
        raise ValueError("chronology_type must be 'standard', 'arstan', or 'residual'.")

    years = np.asarray(chron_data.index, dtype=int)
    idx = chron_data[column].to_numpy(dtype=float)
    depth = chron_data["samp_depth"].to_numpy()
    first_yr, last_yr = int(years.min()), int(years.max())
    site_id = str(header["site_id"])[:6]

    # --- three ITRDB header records ---
    def fld(v, w):                                  # left-justified fixed field
        return str(v)[:w].ljust(w)
    lat, lon = str(header["latitude"]), str(header["longitude"])
    lat_long = (lat + lon) if len(lon) > 5 else (lat + " " + lon)
    yrs = ("%d %d" % (first_yr, last_yr)).ljust(9)
    rec1 = fld(site_id, 6) + "   " + fld(header["site_name"], 52) + fld(header["species_code"], 4)
    rec2 = (fld(site_id, 6) + "   " + fld(header["state_country"], 13)
            + fld(header["species"], 18) + fld(header["elevation"], 5) + "  "
            + fld(lat_long, 10) + "    " + _CRN_TYPE_CODE[ct] + "    " + yrs)
    rec3 = (fld(site_id, 6) + "   " + fld(header["investigators"], 63)
            + fld(header.get("completion_date", ""), 8))
    file.write(rec1 + "\n" + rec2 + "\n" + rec3 + "\n")

    # --- decadal data records ---
    def pair(v, d):
        if np.isnan(v):
            return "9990  0"                        # missing-value marker + 0 depth
        return ("%4d" % int(round(v * 1000))) + ("%3d" % int(d))
    decades = years // 10 * 10
    unique_decades = sorted(set(decades.tolist()))
    lines = []
    for di, dec in enumerate(unique_decades):
        pos = np.flatnonzero(decades == dec)
        n = len(pos)
        pairs = "".join(pair(idx[j], depth[j]) for j in pos)
        # the opening (partial) decade is padded at the FRONT to 10 pairs
        front = ("9990  0" * (10 - n)) if di == 0 else ""
        lines.append(site_id.ljust(6) + ("%4d" % int(years[pos[0]])) + front + pairs)
    # pad the final (partial) decade at the END to 10 pairs
    n_last = int((decades == unique_decades[-1]).sum())
    lines[-1] = lines[-1] + "9990  0" * (10 - n_last)
    file.write("\n".join(lines) + "\n")


def write_txt(data, file):
    header = ["year", "num".rjust(7), "seg".rjust(7), "age".rjust(7), "raw".rjust(7), "std".rjust(7), "res".rjust(7), "ars".rjust(7)]
    file.write("    ".join(header))
    file.write("\n")
    rwi_data = detrend(data, fit="spline", method="ratio", plot=False)
    rwi_chron = chron(rwi_data, prewhiten=False)
    mean_res = chron(rwi_data, biweight=False, prewhiten=False)
    ar_chron = chron(rwi_data, prewhiten=True)

    first = rwi_chron.first_valid_index()
    last = rwi_chron.last_valid_index()

    for year in range(first, last+1):
        samp_dep = rwi_chron["samp_depth"][year]

        # standard chronology of detrended data
        std = rwi_chron["std"][year]

        # residuals of detrended data?
        res =  mean_res["std"][year]

        # residuals of ar modeled data?
        ars = ar_chron["res"][year]
        
        year_data = data.loc[[year]].dropna(axis=1)
        column_names = year_data.columns
        
        seg = 0
        age = 0
        raw = 0
        
        for name in column_names:
            seg += len(data[name].dropna())
            age += year - data[name].first_valid_index() + 1
            raw += data[name][year]
        
        seg = seg/len(column_names)
        age = age/len(column_names)
        raw = raw/len(column_names)

        

        # double check what res and ars are supposed to be
        # work on other columns
        line = [str(year).rjust(4), (f"{samp_dep:.3f}").rjust(7), (f"{seg:.3f}").rjust(7), (f"{age:.3f}").rjust(7), (f"{raw:.3f}").rjust(7), (f"{std:.3f}").rjust(7), (f"{res:.3f}").rjust(7), (f"{ars:.3f}").rjust(7),]
        file.write("    ".join(line))
        file.write("\n")
