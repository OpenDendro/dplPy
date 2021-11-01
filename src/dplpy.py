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
import clipboard
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

# Windows installations
if str(platform.system().lower()) == "windows":
    # Get python runtime version
    version = sys.version_info[0]
    try:
        import pipwin

        if pipwin.__version__ == "0.5.0":
            pass
        else:
            a = subprocess.call(
                "{} -m pip install pipwin==0.5.0".format(sys.executable),
                shell=True,
                stdout=subprocess.PIPE,
            )
            b = subprocess.call(
                "{} -m pip install wheel".format(sys.executable),
                shell=True,
                stdout=subprocess.PIPE,
            )
            subprocess.call("pipwin refresh", shell=True)
        """Check if the pipwin cache is old: useful if you are upgrading porder on windows
        [This section looks if the pipwin cache is older than two weeks]
        """
        home_dir = expanduser("~")
        fullpath = os.path.join(home_dir, ".pipwin")
        file_mod_time = os.stat(fullpath).st_mtime
        if int((time.time() - file_mod_time) / 60) > 90000:
            print("Refreshing your pipwin cache")
            subprocess.call("pipwin refresh", shell=True)
    except ImportError:
        a = subprocess.call(
            "{} -m pip install pipwin==0.5.0".format(sys.executable),
            shell=True,
            stdout=subprocess.PIPE,
        )
        subprocess.call("pipwin refresh", shell=True)
    except Exception as e:
        print(e)
    try:
        import json
    except ImportError:
        try:
           import json
        except ModuleNotFoundError:
            subprocess.call("pipwin install json", shell=True)
    except ModuleNotFoundError or ImportError:
        subprocess.call("pipwin install json", shell=True)
    except Exception as e:
        print(e)
    try:
        import pandas
    except ImportError:
        subprocess.call("pipwin install pandas", shell=True)
    except Exception as e:
        print(e)

# add def files from /src        
from .tucson_json import create_json
from .tucson_var import read_tucson
from .csv_var import read_csv
from .summary_tucson import summary

if str(platform.python_version()) > "3.8":
    from .async_down import downloader
os.chdir(os.path.dirname(os.path.realpath(__file__)))
lpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(lpath)

# Get package version
class Solution:
    def compareVersion(self, version1, version2):
        versions1 = [int(v) for v in version1.split(".")]
        versions2 = [int(v) for v in version2.split(".")]
        for i in range(max(len(versions1), len(versions2))):
            v1 = versions1[i] if i < len(versions1) else 0
            v2 = versions2[i] if i < len(versions2) else 0
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0

ob1 = Solution()

# Open the README
def readme():
    try:
        a = webbrowser.open("https://opendendro.github.io/opendendro/python/", new=2)
        if a == False:
            print("Your computer does not use a monitor, and cannot display the webpages")
    except Exception as e:
        print(e)

def read_from_parser(args):
    readme()

print("")

spacing = "                               "

def main(args=None):
    parser = argparse.ArgumentParser(description="dplPy v0.1")
    subparsers = parser.add_subparsers()

    parser_read = subparsers.add_parser(
        "readme", help="Go to https://opendendro.github.io/opendendro/python/"
    )
    parser_read.set_defaults(func=read_from_parser)

    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
    func(args)


if __name__ == "__main__":
    main()
