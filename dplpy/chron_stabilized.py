from rbar import get_running_rbar, mean_series_intercorrelation
from chron import chron
import numpy as np
import pandas as pd
import warnings


def chron_stabilized(rwi_data: pd.DataFrame, win_length=50, min_seg_ratio=0.33, biweight=True, running_rbar=False):
    """ Variance Stabilization functions
    
    Extended Summary
    ----------------
    Builds a variance stabilized mean-value chronology from a pandas dataframe
    of detrended ring widths, by multiplying the chronology with the square
    root of the effective independent sample size, Neff, defined as 

    Neff = n(t) / 1+(n(t)-1)rbar(t)

        where n(t) is the number of series at time t, and rbar(t) is the
        mean pairwise correlation between all series (not "interseries
        correlation" -- see dpl.interseries_cor() -- which correlates each
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
        as part of chronology output
            
    Returns
    -------
    stabilized_chron: a pandas dataframe of a mean value chronology with stabilized
                      variance.
        
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
    if not isinstance(rwi_data, pd.DataFrame):
        raise TypeError("Expected data input to be a pandas dataframe, not " + str(type(rwi_data)) + ".")
    
    
    num_years = rwi_data.shape[0]

    if win_length > num_years:
        raise ValueError("Window length should not be greater than the number of rows in the dataset")
    
    if min_seg_ratio <= 0 or min_seg_ratio > 1:
        raise ValueError("min_seg_ratio cannot be <= 0 or > 1")
    
    # Matches dplR's chron.stabilized(): these are two independent checks (an
    # absolute floor and a relative ceiling), not one combined 30-50% band --
    # either, both, or neither can fire depending on win_length and num_years.
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

    rbar_array = np.full(zero_mean_data.shape[0], np.nan)

    if win_length % 2 == 0:
        target = (win_length)/2
    else:
        target = (win_length-1)/2
    
    for i in range(num_years-win_length + 1):
        data_segment = zero_mean_data[i:i + win_length]
        if data_segment.shape[0] < win_length:
            continue
        target_index = int(i + target)
        rbar_array[target_index] = get_running_rbar(data_segment, min_seg_ratio)

    rbar_array = pad_rbar_array(rbar_array, n_samps)

    reg_chron = chron(zero_mean_data, biweight=biweight, plot=False)

    mean_rwis = reg_chron["Mean RWI"].to_numpy()
    denom = np.multiply(n_samps-1, rbar_array) + 1

    n_eff = np.minimum(np.divide(n_samps, denom), n_samps)
    # The overall rbar constant is deliberately unmasked (apply_mask=False),
    # unlike the moving-window rbar above -- dplR computes this one as a plain
    # pairwise-complete correlation mean over the whole record, with no
    # minimum-overlap filtering between series pairs.
    rbar_const = mean_series_intercorrelation(zero_mean_data, "pearson", min_seg_ratio, apply_mask=False)
    stabilized_means = np.multiply(mean_rwis, np.sqrt(n_eff * rbar_const))

    if running_rbar:
        stabilized_chron =  pd.DataFrame(data={"Adjusted CRN": stabilized_means + mean_val, "Running rbar": rbar_array, "Sample depth": n_samps}, index=reg_chron.index)
    else:
        stabilized_chron =  pd.DataFrame(data={"Adjusted CRN": stabilized_means + mean_val, "Sample depth": n_samps}, index=reg_chron.index)

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
