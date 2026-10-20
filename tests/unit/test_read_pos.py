import os
import warnings

import numpy as np
import pandas as pd

import dplpy as dpl

POS = os.path.join(os.path.dirname(__file__), "..", "data", "pos")


def _q(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def test_read_pos_single_whole_ring():
    pc = _q(dpl.read_pos, os.path.join(POS, "simple_test.pos"))
    rwl = pc.rwl()
    assert list(rwl.columns) == ["simple_test"]
    assert rwl.attrs.get("kind") == "rwl"
    # widths are positive mm, dated to the outer year inward
    s = rwl["simple_test"].dropna()
    assert (s > 0).all()
    assert int(s.index.max()) == 2022
    att = pc.attributes.set_index("series").loc["simple_test"]
    # computed radius is within 2% of CooRecorder's own (path-sum vs straight line)
    assert abs(att["radius_calc"] - att["radius_file"]) / att["radius_file"] < 0.02
    # recomputed distance to pith matches the file to a few hundredths of a mm
    assert abs(att["distance_to_pith"] - att["distance_to_pith_file"]) < 0.1


def test_multipoint_via_points_excluded():
    # simple_test contains multipoints; if multi2 via-points were counted the
    # radius would overshoot by several mm. The 2% check above guards this, but
    # assert the computed radius is near the file value directly too.
    pc = _q(dpl.read_pos, os.path.join(POS, "simple_test.pos"))
    att = pc.attributes.set_index("series").loc["simple_test"]
    assert att["radius_calc"] < att["radius_file"] + 1.0


def test_seasonwood_split_sums_to_ring():
    pc = _q(dpl.read_pos, os.path.join(POS, "very_simple_seaswood_test.pos"))
    rwl, ew, lw = pc.rwl(), pc.earlywood(), pc.latewood()
    assert ew is not None and lw is not None
    both = pd.DataFrame({"rw": rwl.iloc[:, 0], "ew": ew.iloc[:, 0],
                         "lw": lw.iloc[:, 0]}).dropna()
    assert len(both) > 40
    assert np.allclose(both["ew"] + both["lw"], both["rw"], atol=1e-9)


def test_directory_stacks_and_reports_not_read():
    pc = _q(dpl.read_pos, POS)
    # the six readable fixtures stack into one frame; broken + 7.7 file are skipped
    assert pc.rwl().shape[1] == 6
    nr = set(os.path.basename(f) for f in pc.not_read["file"])
    assert {"broken.pos", "old_file.pos"} <= nr


def test_version_floor_rejects_pre_7_8():
    pc = _q(dpl.read_pos, os.path.join(POS, "old_file.pos"))
    assert pc.rwl() is None
    assert "7.8" in pc.not_read.iloc[0]["reason"]


def test_readers_delegates_pos():
    # single file, directory, and explicit format all return a ring-width frame
    f = _q(dpl.readers, os.path.join(POS, "simple_test.pos"))
    assert isinstance(f, pd.DataFrame) and list(f.columns) == ["simple_test"]
    assert "dplpy_pos_attributes" in f.attrs
    d = _q(dpl.readers, POS)
    assert d.shape[1] == 6
    d2 = _q(dpl.readers, POS, format="pos")
    assert d2.shape[1] == 6


def test_readers_sniffs_pos_by_content(tmp_path):
    # a .pos file under a non-.pos suffix is detected from the DENDRO header
    src = open(os.path.join(POS, "simple_test.pos"), encoding="latin-1").read()
    p = tmp_path / "mystery.txt"
    p.write_text(src, encoding="latin-1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = dpl.readers(str(p))
    assert f.shape[1] == 1


def test_yrs_to_pith_and_feeds_rcs():
    pc = _q(dpl.read_pos, POS)
    rwl = pc.rwl()
    d2 = pc.attributes[["series", "distance_to_pith"]]
    po = _q(dpl.yrs_to_pith, rwl, d2, method="dist")
    assert list(po.columns) == ["series", "d2pith", "mean_rw", "pith_offset"]
    assert str(po["pith_offset"].dtype) == "Int64"
    # a series with a pith estimate feeds rcs as a pith-offset table
    have = po.dropna(subset=["pith_offset"])
    sub = rwl[[c for c in rwl.columns if c in set(have["series"])]]
    r = _q(dpl.rcs, sub, have[["series", "pith_offset"]], biweight=True, plot=False)
    assert isinstance(r, pd.DataFrame) and r.shape[0] == sub.shape[0]


def test_yrs_to_pith_methods_and_missing():
    pc = _q(dpl.read_pos, POS)
    rwl = pc.rwl()
    d2 = pc.attributes[["series", "distance_to_pith"]]
    rings = _q(dpl.yrs_to_pith, rwl, d2, method="rings", n_rings=5)
    assert (rings.dropna(subset=["pith_offset"])["pith_offset"] >= 0).all()
    # a series without a pith distance (muchos_multis has no pith) -> <NA>, warned
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = dpl.yrs_to_pith(rwl, d2)
    mm = out.set_index("series").loc["muchos_multis", "pith_offset"]
    assert pd.isna(mm)
