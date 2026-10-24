__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2026  OpenDendro

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

# Title: pcreg_plot.py
# Description: The figure pcreg draws -- the reconstruction over the full
#   predictor period with the actual climate target overlaid, the calibration
#   and validation windows shaded, and the key skill statistics annotated.
#   Styled per-Axes via _plot_style, like the rest of dplPy's plots.

import numpy as np
import matplotlib.pyplot as plt

from ._plot_style import style_axes, finalize_font

_ACTUAL = "#444444"        # observed target
_RECON = "#1f6f8b"         # reconstruction
_CV = "#e6872e"            # cross-validation (held-out) prediction
_CAL = "#2ca02c"           # calibration shading
_VAL = "#d62728"           # validation shading


def _plot_pcreg(result, target, show=True):
    """Draw the reconstruction with the target overlaid and the windows shaded.

    Parameters
    ----------
    result : PcregResult
        The fitted result (carries the reconstruction, windows, and stats).
    target : pandas.Series
        The year-indexed climate target that was reconstructed.
    show : bool
        Call ``plt.show()`` at the end.

    Returns
    -------
    (fig, ax)
    """
    rec = result.reconstruction
    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Observed target and reconstruction.
    ax.plot(target.index, target.to_numpy(), color=_ACTUAL, lw=1.1,
            label="observed", zorder=3)
    ax.plot(rec.index, rec.to_numpy(), color=_RECON, lw=1.4,
            label="reconstruction", zorder=4)

    # Cross-validation prediction (held-out, calibration period), if present.
    cvr = getattr(result, "cv_reconstruction", None)
    if cvr is not None and cvr.notna().any():
        ax.plot(cvr.index, cvr.to_numpy(), color=_CV, lw=0.0, marker="o",
                ms=3, alpha=0.7, label="cross-validation", zorder=5)

    # Shade the calibration and validation windows.
    if result.calibration is not None:
        c0, c1 = result.calibration
        ax.axvspan(c0, c1, color=_CAL, alpha=0.08, zorder=0)
        ax.text(0.5 * (c0 + c1), ax.get_ylim()[1], "calibration",
                color=_CAL, ha="center", va="top", fontsize=8)
    if result.validation is not None:
        v0, v1 = result.validation
        ax.axvspan(v0, v1, color=_VAL, alpha=0.08, zorder=0)
        ax.text(0.5 * (v0 + v1), ax.get_ylim()[1], "validation",
                color=_VAL, ha="center", va="top", fontsize=8)

    # Annotate the skill statistics.
    bits = ["n$_{pred}$=%d" % len(result.predictors_screened),
            "n$_{PC}$=%d" % result.n_pc,
            "order=%d" % len(result.pcs_selected),
            "cal R$^2$=%.2f" % result.cal_r2]
    s = result.stats
    if s.get("r") is not None and np.isfinite(s.get("r", np.nan)):
        bits.append("ver r=%.2f" % s["r"])
    if s.get("RE") is not None and np.isfinite(s.get("RE", np.nan)):
        bits.append("RE=%.2f" % s["RE"])
    if s.get("CE") is not None and np.isfinite(s.get("CE", np.nan)):
        bits.append("CE=%.2f" % s["CE"])
    if s.get("cv_RE") is not None and np.isfinite(s.get("cv_RE", np.nan)):
        bits.append("CV-RE=%.2f" % s["cv_RE"])
    ax.set_title("   ".join(bits), fontsize=10)

    ax.set_xlabel("Year")
    ax.set_ylabel(target.name or "target")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    style_axes(ax, xgrid=False, ygrid=True)
    finalize_font(fig)
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax
