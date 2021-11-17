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

# Date:2021-11-16
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
from bs4 import BeautifulSoup
import os
import sys
import webbrowser

os.chdir(os.path.dirname(os.path.realpath(__file__)))
lpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(lpath)

# Help Menu
def help():
    try:
        print("*Welcome to the dplPy Help Menu*")
        print("")
        print("....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⁞")
        print("")
        print("README")
        print("")
        print("to view the documentation online from terminal, type:")
        print("$ python src/dplpy.py readme()")
        print("")
        print("from Python Console:")
        print(">>> import dplpy")
        print(">>> dplpy.readme()")
        print("")
        print("or")
        print("to visit our website click url: https://opendendro.org/dplpy")
        print("")
        print("READERS")
        print("")
        print("to import ring width series from termial type:")
        print("$ python src/dplpy.py reader(\"/folder/filename.csv\")")
        print("")
        print("from Python Console:")
        print("")
        print(">>> import dplpy")
        print(">>> dplpy.readers(\"/folder/filename.csv\")")
        print("where input = a single ring width series formatted file")
        print("")
        print("WRITERS")
        print("")
        print("to write or convert outputs to new file from terminal, type:")
        print("$ python src/dplpy.py writer(\"/folder/filename.csv\")")
        print("")
        print("from Python Console:")
        print("")
        print(">>> import dplpy")
        print(">>> dplpy.writers(input=\"/folder/filename.csv\",output=\"/outputfolder/outputfile.rwl\")")
        print("where input = a ring width series file, and output= a ring width series file")
        print("")
        print("SUMMARY STATISTICS")
        print("TBD")
    except Exception as e:
        print(e)
# set the definition for the help function
def help_from_parser(args):
    help()

# Open the Website README (Manual documentation) 
def readme():
    try:
        a = webbrowser.open("https://opendendro.github.io/opendendro/python/", new=2)
        print("Success: Check your web browser for a new tab")
        if a == False:
            print("Your computer does not use a monitor, and cannot display the webpages")
    except Exception as e:
        print(e)
# set the definition for the help function
def readme_from_parser(args):
    readme()

# Get package versioning -- commented out until we add dplPy to pypi.org
#def dplpy_version():
#    url = "https://pypi.org/project/dplpy/"
#    source = requests.get(url)
#    html_content = source.text
#    soup = BeautifulSoup(html_content, "html.parser")
#    company = soup.find("h1")
#    vcheck = ob1.compareVersion(
#        company.string.strip().split(" ")[-1],
#        pkg_resources.get_distribution("dplpy").version,
#    )
#    if vcheck == 1:
#        print(
#            "\n"
#            + "========================================================================="
#        )
#        print(
#            "Current version of dplPy is {} upgrade to lastest version: {}".format(
#                pkg_resources.get_distribution("dplpy").version,
#                company.string.strip().split(" ")[-1],
#            )
#        )
#        print(
#            "========================================================================="
#        )
#    elif vcheck == -1:
#        print(
#            "\n"
#            + "========================================================================="
#        )
#        print(
#            "Possibly running staging code {} compared to pypi release {}".format(
#                pkg_resources.get_distribution("dplpy").version,
#                company.string.strip().split(" ")[-1],
#            )
#        )
#        print(
#            "========================================================================="
#        )
#
#dplpy_version()

# set the definition for the Readers functions (from readers.py)
def readers_from_parser(args):
    readers(input=args.input)

# set the definition for the Writers funcitons (from writers.py)
#def writers_from_parser(args):
#    writers(input=args.input,output=args.output)

# Summary
#def summary_from_parser(args):
#    summary()

# creates whitespace
print("")

spacing = "                               "

# Import definition functions from other files in the src/ path

from readers import readers
# Commenting out extra features until we're ready to implement them
# import writers
# import summary

def main(args=None):
    parser = argparse.ArgumentParser(description="dplPy v0.1") # update version as we update packages
    subparsers = parser.add_subparsers()

# Help Documentation
    parser_help = subparsers.add_parser(
        "help", help="Echo the Help Menu"
    )

    parser_help.set_defaults(func=help_from_parser)

# README Documentation pages
    parser_readme = subparsers.add_parser(
        "readme", help="Goes to the website: https://opendendro.github.io/opendendro/python/"
    )

    parser_readme.set_defaults(func=readme_from_parser)

# Reader file input parser
    parser_readers = subparsers.add_parser(
        "readers", help="Read input ring width series files (.CSV, .JSON, .RWL, .TXT) into an array"
    )

    parser_readers.add_argument(
        "--input",
        help="select a valid input ring width series file (.CSV, .JSON, .RWL, .TXT) ",
    )

    parser_readers.set_defaults(func=readers_from_parser)

# Writer file output
#    parser_writers = subparsers.add_parser(
#        "writer", help="Write out ring width series from array to file (.CSV, .JSON, .RWL, .TXT)"
#    )
#    parser_writers.add_argument(
#        "--input",
#        help="select a valid input ring width series file (.CSV, .JSON, .RWL, .TXT) ",
#    )
#    parser_writers.add_argument(
#        "--output",
#        help="select a valid output ring width series file (.CSV, .JSON, .RWL, .TXT) ",
#    )
#    parser_writers.set_defaults(func=writers_from_parser)

# Summary Statistics
#    parser_summary = subparsers.add_parser(
#        "summary", help="Prints out the summary statistics for an array or input file"
#    )
#
#    parser_summary.set_defaults(func=read_from_parser)

# TBD

# finish argument parsing
    args = parser.parse_args()

    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
    func(args)

if __name__ == "__main__":
    main()