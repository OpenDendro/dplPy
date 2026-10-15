"""Tests for the console entry point (`dplpy` / `python -m dplpy`), the
in-console help menu, readme(), and the readers_url wrapper."""
import io
import runpy
import sys
import warnings
from unittest import mock

import pytest

import dplpy as dpl
from dplpy import cli


# ---------------------------------------------------------------- version ---

def test_version_prefers_source_version():
    # the source tree's __version__ wins over whatever distribution metadata
    # happens to be installed (H3: a stale egg-info reported 0.4.0)
    assert cli._version() == dpl.__version__


def test_version_falls_back_to_metadata_then_placeholder(monkeypatch):
    import importlib

    def boom(*a, **k):
        raise ImportError("no __version__")
    # make `from . import __version__` fail by hiding the attribute
    monkeypatch.delattr(dpl, "__version__")
    importlib.reload(cli)
    try:
        with mock.patch.object(cli, "_dist_version", return_value="9.9.9"):
            assert cli._version() == "9.9.9"
        with mock.patch.object(cli, "_dist_version",
                               side_effect=cli.PackageNotFoundError):
            assert cli._version().startswith("unknown")
    finally:
        monkeypatch.undo()
        importlib.reload(cli)
    assert cli._version() == dpl.__version__


def test_main_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.strip() == "dplpy " + dpl.__version__


def test_main_no_args_prints_banner(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "dplPy " + dpl.__version__ in out
    assert "import dplpy as dpl" in out
    assert "https://opendendro.github.io/dplPy/" in out


def test_python_m_dplpy_runs(capsys):
    # `python -m dplpy --version` goes through __main__.py
    argv = sys.argv
    sys.argv = ["dplpy", "--version"]
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("dplpy", run_name="__main__", alter_sys=True)
    finally:
        sys.argv = argv
    assert exc.value.code == 0
    assert "dplpy " + dpl.__version__ in capsys.readouterr().out


# ------------------------------------------------------------------- help ---

def test_help_overview_lists_every_public_function(capsys):
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    for name in dpl.__all__:
        assert name in out, name
    # generated from the live API: the stale hand-written signature is gone
    assert "writers(input=" not in out
    assert "opendendro.org/python" not in out
    assert "TBD" not in out


def test_dpl_help_no_args_equals_cli_help(capsys):
    dpl.help()
    a = capsys.readouterr().out
    cli.main(["help"])
    b = capsys.readouterr().out
    assert a == b
    assert "READING AND WRITING DATA" in a


def test_help_one_function_shows_live_signature_and_docstring(capsys):
    dpl.help("readers")
    out = capsys.readouterr().out
    assert out.startswith("readers(")
    assert "strict" in out.split("\n")[0]            # a real parameter name
    assert "Parameters" in out                        # the full docstring
    assert "reference/" in out


def test_help_accepts_dpl_prefix_and_callable(capsys):
    dpl.help("dpl.glk")
    by_name = capsys.readouterr().out
    dpl.help(dpl.glk)
    by_obj = capsys.readouterr().out
    assert by_name == by_obj
    assert by_name.startswith("glk(")


def test_help_unknown_name_is_friendly(capsys):
    dpl.help("xdat")
    out = capsys.readouterr().out
    assert "no public function named 'xdat'" in out
    assert "xdate" in out                             # suggestion
    cli.main(["help", "nope_nothing"])
    assert "no public function" in capsys.readouterr().out


def test_help_cli_subcommand_with_name(capsys):
    assert cli.main(["help", "chron"]) == 0
    assert capsys.readouterr().out.startswith("chron(")


def test_no_sphinx_roles_in_cli_or_plot_docstrings():
    import inspect
    from dplpy import plot as plot_mod, _plot_style
    for mod in (cli, plot_mod, _plot_style):
        assert ":func:" not in inspect.getsource(mod), mod.__name__
        assert "opendendro.org/dplpy-man" not in inspect.getsource(mod), mod.__name__


# ----------------------------------------------------------------- readme ---

def test_readme_opens_docs_url(capsys):
    with mock.patch.object(cli.webbrowser, "open", return_value=True) as wb:
        dpl.readme()
    wb.assert_called_once()
    assert wb.call_args[0][0] == "https://opendendro.github.io/dplPy/"
    out = capsys.readouterr().out
    assert "https://opendendro.github.io/dplPy/" in out


def test_readme_without_browser_prints_url(capsys):
    with mock.patch.object(cli.webbrowser, "open", return_value=False):
        cli.main(["readme"])
    out = capsys.readouterr().out
    assert "https://opendendro.github.io/dplPy/" in out
    assert "No web browser" in out


# ------------------------------------------------------------ readers_url ---

def test_readers_url_forwards_to_readers(DATA):
    # readers_url is a thin wrapper over readers(); feed it a local file's
    # bytes through a mocked urlopen so no network is needed.
    rwl_path = DATA / "rwl" / "ca533.rwl"
    payload = rwl_path.read_bytes()

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

        def info(self):
            return {}

    url = "https://example.invalid/ca533.rwl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = dpl.readers(str(rwl_path))
        # readers() calls urllib.request.urlopen(url, timeout=...) by module path
        with mock.patch("urllib.request.urlopen", return_value=_Resp(payload)) as uo:
            got = dpl.readers_url(url)
    assert uo.call_args[0][0] == url
    assert list(got.columns) == list(expected.columns)
    assert got.shape == expected.shape
