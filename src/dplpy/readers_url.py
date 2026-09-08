from .readers import readers


def readers_url(url, header=None, skip_lines=0, strict=True):
    """Read a Tucson (.rwl) file directly from a URL.

    Deprecated: ``readers()`` now accepts an http(s) URL directly, so
    ``dpl.readers(url)`` does the same thing. This thin wrapper is kept for
    backward compatibility and simply forwards to ``readers``.
    """
    return readers(url, skip_lines=skip_lines, header=header, strict=strict)
