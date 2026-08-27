from __future__ import print_function


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

# Date: 5/27/2022
# Author: Tyson Lee Swetnam
# Project: OpenDendro dplPy
# Description: Imports main functionality for package
# example usage from Python Console:
# >>> import dplpy as dpl
# >>> dpl.readme()
# >>> dpl.help()
# >>> dpl.readers(input="tests/csv/ca533.csv")
# >>> dpl.writers(input="tests/csv/ca533.csv",output="ca533.rwl")

import argparse
import webbrowser
from importlib.metadata import version as _dist_version, PackageNotFoundError


def _version():
    """The installed package version (via importlib.metadata), or a clear
    placeholder when running from a source tree that isn't installed."""
    try:
        return _dist_version("dplpy")
    except PackageNotFoundError:
        return "unknown (running from an uninstalled source tree)"


# Help Menu
def help():
    try:
        print("*Welcome to the dplPy Help Menu*")
        print("")
        print("....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⁞")
        print("")
        print("README \n")
        print("to view the documentation online:")
        print(">>> import dplpy as dpl")
        print(">>> dpl.readme() \n")
        print("or visit our website click url: https://opendendro.org/python \n")
        print("READERS \n")
        print("Import ring width series from the Python Console: \n")
        print(">>> import dplpy as dpl")
        print(">>> dpl.readers(\"/folder/filename.csv\") \n")
        print("WRITERS \n")
        print("Write or convert outputs to a new file from the Python Console: \n")
        print(">>> import dplpy as dpl")
        print(">>> dpl.writers(input=\"/folder/filename.csv\",output=\"/outputfolder/outputfile.rwl\") \n")
        print("SUMMARY STATISTICS")
        print("TBD")
        print("")
        print("")
        print("END HELP MANUAL")
    except Exception as e:
        print(e)

# Open the Website README (Manual documentation)
def readme():
    try:
        a = webbrowser.open("https://opendendro.org/dplpy-man/", new=2)
        print("Success: Check your web browser for a new tab")
        if a is False:
            print("Your computer does not use a monitor, and cannot display the webpages")
    except Exception as e:
        print(e)

# Console entry point (the `dplpy` command and `python -m dplpy`).
def main(argv=None):
    """Small convenience CLI. dplPy is primarily used as a library
    (``import dplpy as dpl``); this exposes version / help / manual from a shell."""
    parser = argparse.ArgumentParser(
        prog="dplpy",
        description="dplPy -- the Dendrochronology Program Library for Python. "
                    "Primarily used as a library: `import dplpy as dpl`.",
    )
    parser.add_argument("--version", action="version", version="dplpy " + _version())
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("help", help="print the console help menu")
    sub.add_parser("readme", help="open the online manual in a web browser")
    args = parser.parse_args(argv)

    if args.command == "help":
        help()
    elif args.command == "readme":
        readme()
    else:                                   # no subcommand: short banner
        print("dplPy " + _version())
        print("The Dendrochronology Program Library for Python.")
        print("")
        print("Use it as a library:  import dplpy as dpl")
        print("Console help:         dplpy help    (or dpl.help() in Python)")
        print("Online manual:        dplpy readme  (or dpl.readme() in Python)")
    return 0
