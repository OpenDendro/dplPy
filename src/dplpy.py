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

# Date:2021-10-31
# Author: Tyson Lee Swetnam
# Project: OpenDendro dplPy
# Description: Imports main functionality for package
# example usage:

import argparse
import base64
from bs4 import BeautifulSoup
import datetime
import dateutil.parser
import json
import os
import pkg_resources
import platform
import requests
import subprocess
import sys
import time
import webbrowser

os.chdir(os.path.dirname(os.path.realpath(__file__)))
lpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(lpath)

# Opens the Python README website
def readme():
    try:
        a = webbrowser.open("https://opendendro.github.io/opendendro/python/", new=2)
        if a == False:
            print("Your computer does not use a monitor, and cannot display the webpages")
    except Exception as e:
        print(e)

def read_from_parser(args):
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

# Commenting out until we've corrected the read_rwl.py read_csv.py and summary.py 
# from .read_rwl import read_rwl
# from .read_csv import read_csv
# from .summary import summary_rwl
# from .summary import summary_csv

def main(args=None):
    parser = argparse.ArgumentParser(description="dplPy v0.1") # update version as we update packages
    subparsers = parser.add_subparsers()

    parser_read = subparsers.add_parser(
        "readme", help="Goes to the website: https://opendendro.github.io/opendendro/python/"
    )
    
    parser_read.set_defaults(func=read_from_parser)
    args = parser.parse_args()

    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
    func(args)

if __name__ == "__main__":
    main()