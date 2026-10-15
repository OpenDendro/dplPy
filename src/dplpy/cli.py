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

# Date: 5/27/2022 (help menu regenerated from the live API 2026-10)
# Author: Tyson Lee Swetnam
# Project: OpenDendro dplPy
# Description: console helpers -- the `dplpy` command / `python -m dplpy`, the
#   in-console help menu, and a shortcut that opens the online manual.
# example usage from Python Console:
# >>> import dplpy as dpl
# >>> dpl.help()              # overview of every public function
# >>> dpl.help("readers")     # signature + docstring of one function
# >>> dpl.readme()            # open the online manual

import argparse
import inspect
import webbrowser
from importlib.metadata import version as _dist_version, PackageNotFoundError

DOCS_URL = "https://opendendro.github.io/dplPy/"

# Rough grouping for the help overview. Names that are not listed here (or that
# are added to dplpy.__all__ later) fall into "Other", and names listed here that
# no longer exist are skipped -- so the menu is always generated from the live
# package and cannot describe functions that are not there.
_GROUPS = [
    ("Reading and writing data",
     ["readers", "readers_url", "metadata", "read_crn", "read_ids", "writers",
      "read_tridas", "write_tridas", "to_lipd", "from_lipd", "lipd_metadata",
      "combine_rwl", "edit_rwl", "fill_internal", "SiteMetadata",
      "TridasProject", "TridasMetadata"]),
    ("Describing and checking data",
     ["summary", "stats", "report", "check", "samp_stats", "plot",
      "common_interval"]),
    ("Detrending and standardization",
     ["detrend", "rcs", "sfrcs", "mrcs", "sfmrcs", "ssf", "ads", "powt",
      "tree_mean", "bai_out", "bai_in", "po_to_wc", "wc_to_po"]),
    ("Autoregressive modeling",
     ["ar_func", "autoreg"]),
    ("Chronologies",
     ["chron", "chron_stabilized", "stabilize_chron", "chron_ars"]),
    ("Crossdating",
     ["xdate", "xdate_report", "xdate_plot", "xdate_floater", "xdate_undated",
      "series_corr", "interseries_corr"]),
    ("Statistics and agreement",
     ["rwi_stats", "rwi_stats_running", "sss", "glk", "sgc", "sens1", "sens2"]),
    ("Help",
     ["help", "readme"]),
]


def _version():
    """The dplPy version: ``dplpy.__version__`` from the source tree first (so a
    source checkout reports its own number), then the installed distribution's
    metadata, then a clear placeholder."""
    try:
        from . import __version__
        return __version__
    except Exception:
        pass
    try:
        return _dist_version("dplpy")
    except PackageNotFoundError:
        return "unknown (running from an uninstalled source tree)"


def _public_api():
    """name -> object for every name in ``dplpy.__all__`` (lazy import: cli is
    imported by the package itself, so the package cannot be imported here at
    module level)."""
    import dplpy as _dpl
    return {name: getattr(_dpl, name) for name in _dpl.__all__
            if hasattr(_dpl, name)}


def _signature(obj):
    """``name(arg, kw=default, ...)`` from the live object, or just the name."""
    name = getattr(obj, "__name__", str(obj))
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return name + "(...)"
    # drop type annotations: the docstring explains the types more readably
    params = [p.replace(annotation=inspect.Parameter.empty)
              for p in sig.parameters.values()]
    sig = sig.replace(parameters=params,
                      return_annotation=inspect.Signature.empty)
    return name + str(sig)


def _first_paragraph(obj):
    """The first paragraph of an object's docstring, joined onto one line."""
    doc = inspect.getdoc(obj) or ""
    para = doc.split("\n\n", 1)[0]
    return " ".join(line.strip() for line in para.splitlines() if line.strip())


def _overview():
    api = _public_api()
    lines = ["dplPy " + _version() + " -- the Dendrochronology Program Library for Python",
             "",
             "Use it as a library:   import dplpy as dpl",
             "Help on one function:  dpl.help(\"readers\")   (or help(dpl.readers))",
             "Online manual:         dpl.readme()   -> " + DOCS_URL,
             ""]
    seen = set()
    for title, names in _GROUPS:
        members = [n for n in names if n in api]
        if not members:
            continue
        lines.append(title.upper())
        for n in members:
            seen.add(n)
            lines.append("  %-20s %s" % (n, _first_paragraph(api[n])))
        lines.append("")
    rest = sorted(n for n in api if n not in seen)
    if rest:
        lines.append("OTHER")
        for n in rest:
            lines.append("  %-20s %s" % (n, _first_paragraph(api[n])))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _describe(name):
    api = _public_api()
    if callable(name) and not isinstance(name, str):
        obj = name
        label = getattr(obj, "__name__", str(obj))
    else:
        label = str(name).strip()
        if label.startswith("dpl."):
            label = label[4:]
        if label not in api:
            close = sorted(n for n in api if label.lower() in n.lower())
            hint = ("  Did you mean: " + ", ".join(close) + "?") if close else ""
            return ("dplPy has no public function named %r.%s\n"
                    "Call dpl.help() with no arguments for the full list.\n"
                    % (label, hint))
        obj = api[label]
    text = [_signature(obj), ""]
    doc = inspect.getdoc(obj)
    text.append(doc if doc else "(no docstring)")
    text.append("")
    text.append("Online reference: " + DOCS_URL + "reference/")
    return "\n".join(text) + "\n"


def help(name=None):
    """Print help for dplPy in the console.

    With no argument, print an overview of every public function (grouped by
    task, each with the first line of its docstring). With a function name --
    ``dpl.help("readers")`` -- or the function itself -- ``dpl.help(dpl.readers)``
    -- print that function's live signature and full docstring. The listing is
    generated from the installed package, so it always matches the real API.

    Parameters
    ----------
    name : str or callable, optional
        A public dplPy function name (``"readers"``, ``"dpl.readers"``) or the
        function object. Omit for the overview.

    Examples
    --------
    >>> dpl.help()
    >>> dpl.help("xdate")
    """
    print(_overview() if name is None else _describe(name))


def readme():
    """Open the online dplPy manual (https://opendendro.github.io/dplPy/) in a
    web browser. Prints the URL as well, for machines without a display."""
    print("dplPy manual: " + DOCS_URL)
    try:
        opened = webbrowser.open(DOCS_URL, new=2)
    except Exception as e:                      # pragma: no cover - browser quirks
        print("Could not open a web browser (%s); use the URL above." % (e,))
        return
    if opened:
        print("Opened in your web browser.")
    else:
        print("No web browser is available on this machine; use the URL above.")


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
    p_help = sub.add_parser("help", help="print the console help menu, or help "
                                         "on one function")
    p_help.add_argument("name", nargs="?", default=None,
                        help="a dplPy function name, e.g. readers")
    sub.add_parser("readme", help="open the online manual in a web browser")
    args = parser.parse_args(argv)

    if args.command == "help":
        help(args.name)
    elif args.command == "readme":
        readme()
    else:                                   # no subcommand: short banner
        print("dplPy " + _version())
        print("The Dendrochronology Program Library for Python.")
        print("")
        print("Use it as a library:  import dplpy as dpl")
        print("Console help:         dplpy help [function]  (or dpl.help() in Python)")
        print("Online manual:        dplpy readme           (or dpl.readme() in Python)")
        print("                      " + DOCS_URL)
    return 0
