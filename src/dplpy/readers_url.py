import os
import urllib.request

from .readers import _lines_to_dataframe


def readers_url(url, header=None, skip_lines=0, on_error="raise"):
    """Read a Tucson (.rwl) file directly from a URL into a Year-indexed dataframe.

    Uses the same parsing pipeline as ``readers`` (header auto-detection, the
    hardened Tucson parser, and -- with on_error="warn" -- salvage mode), so a
    remote file behaves exactly like a local one.
    """
    if on_error not in ("raise", "warn"):
        raise ValueError("on_error must be 'raise' or 'warn'")

    FORMAT = "." + url.split(".")[-1]
    raw_lines = urllib.request.urlopen(url).read().decode("utf-8").split("\n")

    df = _lines_to_dataframe(raw_lines, skip_lines, header, on_error,
                             os.path.basename(url))
    if df is None:
        if on_error == "warn":
            import warnings
            warnings.warn("No usable data could be read from " + url
                          + "; returning None (on_error='warn').")
            return None
        raise ValueError(
            "Error reading file. Check that the URL is correct and that the file "
            "formatting is consistent with " + FORMAT + " format."
        )

    salvage_report = df.attrs.get("dplpy_salvage", [])
    df.set_index("Year", inplace=True, drop=True)
    df.attrs["dplpy_salvage"] = salvage_report

    print("\nSUCCESS!\nFile read as:", FORMAT, "file\n")
    print("Series names:")
    print(list(df.columns), "\n")
    return df
