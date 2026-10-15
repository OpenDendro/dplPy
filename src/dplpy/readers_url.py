from .readers import readers
from ._validate import _warn


def readers_url(url, **kwargs):
    """Read a Tucson (.rwl) file directly from a URL.

    Deprecated: ``readers()`` accepts an http(s) URL directly, so
    ``dpl.readers(url)`` does the same thing. This thin wrapper is kept for
    backward compatibility: it emits a ``DeprecationWarning`` and forwards every
    keyword (``header``, ``skip_lines``, ``strict``, ``join``, ``encoding``,
    ``format``, ``verbose``) to ``readers``.
    """
    _warn("readers_url() is deprecated and will be removed in a future release; "
          "dpl.readers(url) reads a URL directly.", DeprecationWarning)
    return readers(url, **kwargs)
