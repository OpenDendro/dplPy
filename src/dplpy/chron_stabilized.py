from .rbar import get_running_rbar, mean_series_intercorrelation, pairwise_corr_mean
from .chron import chron
from .smoothingspline import spline, variance_stabilize_spline
import numpy as np
import pandas as pd
from ._validate import _require_dataframe
import warnings


def chron_stabilized(rwi_data: pd.DataFrame, win_length=50, min_seg_ratio=1/3,
                     biweight=True, running_rbar=False, method="running_rbar",
                     spline_nyrs=None):
    """Build a variance-stabilized mean-value chronology.

    Scales a mean-value chronology by the square root of the effective independent
    sample size Neff = n(t) / (1 + (n(t)-1) rbar(t)), where n(t) is the sample depth
    and rbar(t) the mean pairwise interseries correlation, computed over windows of
    length ``win_length``. Port of dplR's chron.stabilized().

    Parameters
    ----------
    rwi_data : pd.DataFrame
        detrended ring-width indices.
    win_length : int, default 50
        window length over which rbar values are calculated.
    min_seg_ratio : float, default 1/3
        minimum ratio of non-NA values to window length for a series to count in an
        Neff calculation (1/3 matches dplR's chron.stabilized()).
    biweight : boolean, default True
        average with Tukey's biweight robust mean if True, else the arithmetic mean.
    running_rbar : boolean, default False
        also return the running rbar values in the output (only for
        method='running_rbar').
    method : str, default 'running_rbar'
        the variance-stabilization method:

        - 'running_rbar' : running-window-rbar effective-signal adjustment (Frank et
          al. 2006 "RUNNINGr"); the default. Uses win_length and min_seg_ratio.
        - 'briffa' (alias 'mean_rbar') : the same adjustment with a single
          time-constant rbar over all pairwise overlaps longer than 20 years (Osborn
          et al. 1997; Frank et al. 2006 "MEANr"). Corrects for changing sample size
          only.
        - 'spline' : ARSTAN's ad-hoc spline stabilization, removing any time trend in
          the absolute departures with a smoothing spline. Can remove real
          low-frequency variance -- use with care. Stiffness set by spline_nyrs.
    spline_nyrs : int, float or None, default None
        stiffness for method='spline'. An int is a fixed spline wavelength in years;
        a float in (0, 1) is a fraction of the chronology length; None uses 0.5.
        Ignored by the rbar methods.

    Returns
    -------
    stabilized_chron : pandas dataframe
        a variance-stabilized mean-value chronology indexed by year, with columns
        ``vsc``, ``Running rbar`` (only when running_rbar=True), and ``samp_depth``.

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.chron_stabilized(data, win_length=60, min_seg_ratio=0.4)

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/chron.stabilized.html
    """
    _require_dataframe(rwi_data)
    
    
    if method == "mean_rbar":
        method = "briffa"
    if method not in ("running_rbar", "briffa", "spline"):
        raise ValueError("method must be 'running_rbar', 'briffa' (alias "
                         "'mean_rbar'), or 'spline'; got '" + str(method) + "'.")

    num_years = rwi_data.shape[0]

    if win_length > num_years:
        raise ValueError("Window length should not be greater than the number of rows in the dataset")

    if min_seg_ratio <= 0 or min_seg_ratio > 1:
        raise ValueError("min_seg_ratio cannot be <= 0 or > 1")

    # win_length only drives the running-window rbar; warn about it only there.
    # Matches dplR's chron.stabilized(): these are two independent checks (an
    # absolute floor and a relative ceiling), not one combined 30-50% band --
    # either, both, or neither can fire depending on win_length and num_years.
    if method == "running_rbar":
        if win_length <= 30:
            warnings.warn("Window length less than 30 is not recommended. Consider a longer window.\n")
        if win_length / num_years > 0.5:
            warnings.warn("Window length greater than 50% of the chronology length is not recommended. Consider a shorter window.\n")
    
    print("Generating variance stabilized chronology...\n")

    # give rbar function a range of years (window length) to calculate rbar for
    # calculate rbar for that window, using either osborn's or frank's or 67spline
    # get rbar for each relevant segment of the dataframe

    # Center on the mean of the per-year (cross-series) means, matching dplR's
    # mean(rowMeans(x)). This is NOT the same as the mean of the per-series
    # means whenever series have staggered start/end years (the normal case
    # for ring-width data) -- and since this constant is multiplied through by
    # the stabilization factor and added back at the end, getting it right
    # matters for the final values, not just as an intermediate detail.
    mean_val = rwi_data.mean(axis=1).mean()

    zero_mean_data = rwi_data - mean_val

    # Sample depth per year, computed directly from the (zero-mean) data --
    # matches dplR's nSamps <- rowSums(!is.na(x0)). Deriving this straight from
    # the data, rather than from chron()'s output, keeps it guaranteed to be
    # the same length as rbar_array below even in the edge case of a year with
    # zero total sample depth across every series.
    n_samps = zero_mean_data.notnull().sum(axis=1).to_numpy()

    reg_chron = chron(zero_mean_data, biweight=biweight, plot=False)
    mean_rwis = reg_chron["std"].to_numpy()

    rbar_array = None
    if method == "running_rbar":
        # RUNNINGr: moving-window rbar (Frank et al. 2006), a port of dplR's
        # chron.stabilized(). rbar varies through time.
        rbar_array = np.full(zero_mean_data.shape[0], np.nan)
        target = (win_length) / 2 if win_length % 2 == 0 else (win_length - 1) / 2
        for i in range(num_years - win_length + 1):
            data_segment = zero_mean_data[i:i + win_length]
            if data_segment.shape[0] < win_length:
                continue
            rbar_array[int(i + target)] = get_running_rbar(data_segment, min_seg_ratio)
        rbar_array = pad_rbar_array(rbar_array, n_samps)
        denom = np.multiply(n_samps - 1, rbar_array) + 1
        n_eff = np.minimum(np.divide(n_samps, denom), n_samps)
        # The overall rbar constant is deliberately unmasked (apply_mask=False),
        # unlike the moving-window rbar above -- dplR computes this one as a plain
        # pairwise-complete correlation mean over the whole record, with no
        # minimum-overlap filtering between series pairs.
        rbar_const = mean_series_intercorrelation(zero_mean_data, "pearson", min_seg_ratio, apply_mask=False)
        stabilized_means = np.multiply(mean_rwis, np.sqrt(n_eff * rbar_const))
        vsc = stabilized_means + mean_val

    elif method == "briffa":
        # MEANr / Osborn constant-rbar / ARSTAN Briffa: a single rbar over all
        # pairwise overlaps longer than 20 years; sample-size correction only.
        rbar_b = _briffa_rbar(zero_mean_data, min_overlap=20)
        denom = np.multiply(n_samps - 1, rbar_b) + 1
        n_eff = np.minimum(np.divide(n_samps, denom), n_samps)
        stabilized_means = np.multiply(mean_rwis, np.sqrt(n_eff * rbar_b))
        vsc = stabilized_means + mean_val

    else:  # method == "spline"
        # ARSTAN's ad-hoc spline variance stabilization: flatten the time trend
        # in the chronology's absolute departures with a smoothing spline.
        vsc = _spline_stabilize(mean_rwis + mean_val, n_samps, spline_nyrs)

    out = {"vsc": vsc, "samp_depth": n_samps}
    if method == "running_rbar" and running_rbar:
        out = {"vsc": vsc, "Running rbar": rbar_array, "samp_depth": n_samps}
    stabilized_chron = pd.DataFrame(data=out, index=reg_chron.index)

    print("SUCCESS!\n")
    return stabilized_chron
    
def pad_rbar_array(rbar_array, n_samps):
    # Fill the leading/trailing runs of not-yet-computed years (the moving
    # window can't center on the first/last win_length/2 years) with the
    # nearest computed value -- matches dplR's approach of padding with the
    # first/last non-NA entry of movingRbarVec, rather than a fixed constant.
    # This requires rbar_array to start as NaN (not 0): a genuinely computed
    # rbar of exactly 0.0 is a valid value, not a "not yet set" placeholder,
    # and treating it as one (as an earlier version of this function did, by
    # initializing with zeros and checking `val != 0`) would search right past
    # a legitimate zero-correlation result.
    valid_positions = np.flatnonzero(~np.isnan(rbar_array))
    if valid_positions.size > 0:
        first_valid = valid_positions[0]
        last_valid = valid_positions[-1]
        rbar_array[:first_valid] = rbar_array[first_valid]
        rbar_array[last_valid:] = rbar_array[last_valid]

    # Matches dplR's movingRbarVec[nSamps==0] <- NA: any year with zero total
    # sample depth across every series has no meaningful rbar, padded or not.
    rbar_array[n_samps == 0] = np.nan

    return rbar_array


def _briffa_rbar(data, min_overlap=20):
    """A single time-constant rbar: the mean of the pairwise correlations between
    series, counting only pairs whose overlap exceeds ``min_overlap`` years.
    """
    return pairwise_corr_mean(data, "pearson", min_overlap=min_overlap, strict=True)


def _spline_stabilize(chron_series, n_samps, spline_nyrs):
    """ARSTAN's ad-hoc spline variance stabilization: centre the chronology, fit a
    smoothing spline to the absolute departures to capture their time trend, divide
    the departures by that trend (restoring sign), then rescale to the chronology's
    original mean and standard deviation. Can remove real low-frequency variance.
    Resolves the ``spline_nyrs`` stiffness convention and clips negatives.
    """
    n = len(chron_series)
    # stiffness: int -> fixed wavelength in years; 0<float<1 -> fraction of n
    if spline_nyrs is None:
        nyrs = max(int(round(0.5 * n)), 2)
    elif isinstance(spline_nyrs, float) and 0 < spline_nyrs < 1:
        nyrs = max(int(round(spline_nyrs * n)), 2)
    else:
        nyrs = max(int(spline_nyrs), 2)
    return variance_stabilize_spline(chron_series, period=nyrs, clip_negative=True,
                                     ok=(np.asarray(n_samps) > 0))
