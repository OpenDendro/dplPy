from .rbar import get_running_rbar, mean_series_intercorrelation, pairwise_corr_mean
from .chron import chron
from .smoothingspline import spline
import numpy as np
import pandas as pd
from ._validate import _require_dataframe
import warnings


def chron_stabilized(rwi_data: pd.DataFrame, win_length=50, min_seg_ratio=0.33,
                     biweight=True, running_rbar=False, method="running_rbar",
                     spline_nyrs=None):
    """ Variance Stabilization functions
    
    Extended Summary
    ----------------
    Builds a variance stabilized mean-value chronology from a pandas dataframe
    of detrended ring widths, by multiplying the chronology with the square
    root of the effective independent sample size, Neff, defined as 

    Neff = n(t) / 1+(n(t)-1)rbar(t)

        where n(t) is the number of series at time t, and rbar(t) is the
        mean pairwise correlation between all series (not "interseries
        correlation" -- see dpl.interseries_corr() -- which correlates each
        series against a composite chronology of the others, rather than
        against each other series individually).

    In the limiting cases, when the rbar is zero or unity, Neff obtains 
    values of the true sample size and unity, respectively.
    Neff is calculated over different segments of the data of
    length `win_length`, and only series with at least `min_seg_ratio` of
    valid values in the segment are considered.

    This is a port of dplR's chron.stabilized() (see reference below); the
    intent is to match that implementation's behavior as closely as possible,
    including its two independent window-length recommendations and its use
    of an unfiltered overall rbar constant (as opposed to the
    windowed rbar, which does apply the min_seg_ratio overlap filter).

    Parameters
    ----------
    rwi_data : pd.DataFrame
        a Pandas dataset representing detrended tree rings/widths.
    win_length : int, default 50
        an integer for specifying the window lengths where rbar values
        will be calculated.
    min_seg_ratio : float, default 0.33
        the minimum ratio of non-NA values to the window length for a series to be
        considered in an Neff calculation.
    biweight : boolean, default True
        flag indicating whether or not to use Tukey's bi-weight robust mean when
        calculating the mean-value chronology
    running_rbar : boolean, default False
        flag indicating whether or not to return the running rbar values
        as part of chronology output (only for method='running_rbar').
    method : str, default 'running_rbar'
        the variance-stabilization method:

        - 'running_rbar' : the running-window-rbar effective-signal adjustment
          (Frank et al. 2006 "RUNNINGr"), a port of dplR's chron.stabilized().
          The default; uses win_length and min_seg_ratio.
        - 'briffa' (alias 'mean_rbar') : the same effective-sample-size adjustment
          but with a single, time-constant rbar computed over all pairwise series
          overlaps longer than 20 years (Osborn et al. 1997 constant-rbar; Frank
          et al. 2006 "MEANr"; ARSTAN's Briffa option). Corrects for changing
          sample size ONLY.
        - 'spline' : ARSTAN's ad-hoc spline stabilization. Removes any time trend
          in the ABSOLUTE departures of the chronology with a smoothing spline,
          so it can stabilize variance even when the heteroscedasticity is NOT a
          sample-size effect. NOTE (per ARSTAN and Osborn et al. 1997): this is
          strictly ad hoc and can remove real low-frequency variance -- use with
          care. Stiffness set by spline_nyrs.
    spline_nyrs : int, float or None, default None
        stiffness for method='spline'. An int is a fixed spline wavelength in
        years (ARSTAN's fixed-n cutoff); a float in (0, 1) is a fraction of the
        chronology length (ARSTAN's %-n cutoff). None uses 0.5 (half the length).
        Ignored by the rbar methods.

    Returns
    -------
    stabilized_chron: a pandas dataframe of a variance-stabilized mean-value
                      chronology, indexed by year. Columns: ``vsc`` (the
                      variance-stabilized chronology, matching dplR's naming),
                      ``Running rbar`` (only when running_rbar=True), and
                      ``samp_depth``.
        
    Examples
    --------
    >>> import dplpy as dpl 
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.chron_stabilized(data, win_length=60, min_seg_ratio=0.4) -> returns mean
                                            value chronology with stabilized
                                            variance.
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
        # ARSTAN's ad-hoc spline stabilization (stabit): flatten the time trend
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
    """A single time-constant rbar: the mean of the pairwise correlations
    between series, counting only pairs whose overlap exceeds ``min_overlap``
    years (ARSTAN's Briffa method uses n > 20). Osborn constant-rbar / Frank
    MEANr."""
    return pairwise_corr_mean(data, "pearson", min_overlap=min_overlap, strict=True)


def _spline_stabilize(chron_series, n_samps, spline_nyrs):
    """ARSTAN's ad-hoc spline variance stabilization (subroutine stabit): centre
    the chronology, fit a smoothing spline to the ABSOLUTE departures to capture
    their time trend, divide the departures by that trend (restoring sign), then
    rescale to the chronology's original mean and standard deviation. Removes
    time-varying variance whatever its cause -- strictly ad hoc (it can remove
    real low-frequency variance), per ARSTAN and Osborn et al. (1997)."""
    tr = np.asarray(chron_series, dtype=float).copy()
    n = len(tr)
    ok = np.asarray(n_samps) > 0
    xbar1 = np.mean(tr[ok])
    sig1 = np.std(tr[ok], ddof=1)
    dep = tr - xbar1
    sign = np.where(dep < 0, -1.0, 1.0)
    absdep = np.abs(dep)
    # stiffness: int -> fixed wavelength in years; 0<float<1 -> fraction of n
    if spline_nyrs is None:
        nyrs = max(int(round(0.5 * n)), 2)
    elif isinstance(spline_nyrs, float) and 0 < spline_nyrs < 1:
        nyrs = max(int(round(spline_nyrs * n)), 2)
    else:
        nyrs = max(int(spline_nyrs), 2)
    x = np.arange(1, n + 1)
    cv = np.asarray(spline(x, absdep, period=nyrs), dtype=float)
    cv = np.where(cv <= 0, np.nan, cv)          # guard: |departures| envelope > 0
    sb = (absdep / cv) * sign
    with np.errstate(invalid="ignore"):
        xbar = np.nanmean(sb[ok])
        sig = np.nanstd(sb[ok], ddof=1)
    sb = ((sb - xbar) / sig) * sig1 + xbar1
    sb[sb < 0] = 0.0
    return sb
