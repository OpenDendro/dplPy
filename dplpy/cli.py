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

import webbrowser

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
