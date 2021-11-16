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
        print("....:....:....:....:....⋮....:....:....:....:....:....⋮")
        print("")
        print("README")
        print("to view the documentation online, type:")
        print("$ python src/dplpy.py readme()")
        print("or")
        print("visit our website here: https://opendendro.org/dplpy")
        print("")
        print("READERS")
        print("to import ring width series data type:")
        print("$ python src/dplpy.py reader(input)")
        print("where input = a single ring width series formatted file")
        print("")
        print("WRITERS")
        print("to write or convert outputs to new file, type:")
        print("$ python src/dplpy.py writer(input,output)")
        print("where input = a ring width series file, and output= a ring width series file")
        print("")
        print("SUMMARY STATISTICS")
        print("TBD")
    except Exception as e:
        print(e)

# set definition for help function
def help_from_parser(args):
    help()

# Open the Website with its README 
def readme():
    try:
        a = webbrowser.open("https://opendendro.github.io/opendendro/python/", new=2)
        if a == False:
            print("Your computer does not use a monitor, and cannot display the webpages")
    except Exception as e:
        print(e)

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

# creates whitespace
print("")

spacing = "                               "

# Commenting out extra features until we're ready to implement them
import readers
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

# Reader file input
    parser_readers = subparsers.add_parser(
        "reader", help="Read input ring width series files (.CSV, .JSON, .RWL, .TXT) into an array"
    )

    parser_readers.set_defaults(func=readers)

# Writer file output
#    parser_writers = subparsers.add_parser(
#        "writer", help="Write out ring width series from array to file (.CSV, .JSON, .RWL, .TXT)"
#    )
#
#    parser_writers.set_defaults(func=writers)

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