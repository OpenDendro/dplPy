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

        where n(t) is the number of series at time t, and rbar is the 
        interseries correlation. 

    In the limiting cases, when the rbar is zero or unity, Neff obtains 
    values of the true sample size and unity, respectively.
    Neff is calculated over different segments of the data of
    length `win_length`, and only series with at least `min_seg_ratio` of
    valid values in the segment are considered.


    Parameters
    ----------
    rwi_data : pd.DataFrame
        a Pandas dataset representing detrended tree rings/widths.
    win_length : int, default 50
        an integer for specifying the window lengths where interseries correlations
        will be calculated.
    min_seg_ratio : float, default 0.33
        the minimum ratio of non-NA values to the window length for a series to be
        considered in an Neff calculation.
    biweight : boolean, default True
        flag indicating whether or not to use Tukey's bi-weight robust mean when
        calculating the mean-value chronology
    running_rbar : boolean, default False
        flag indicating whether or not to return the running interseries
        correlations as part of chronology output
            
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
    
    if win_length < 0.3*num_years or win_length >= 0.5*num_years:
        warnings.warn("We recommend using a window length greater than 30%% but less than 50%% of the chronology length\n")
    
    print("Generating variance stabilized chronology...\n")

    # give rbar function a range of years (window length) to calculate rbar for
    # calculate rbar for that window, using either osborn's or frank's or 67spline
    # get rbar for each relevant segment of the dataframe


    mean_val = rwi_data.mean().mean()

    zero_mean_data = rwi_data - mean_val

    rbar_array = np.zeros(zero_mean_data.shape[0])

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

    rbar_array = pad_rbar_array(rbar_array)

    reg_chron = chron(zero_mean_data, biweight=biweight, plot=False)

    mean_rwis = reg_chron["Mean RWI"].to_numpy()
    samp_deps = reg_chron["Sample depth"].to_numpy()
    denom = np.multiply(samp_deps-1, rbar_array) + 1

    n_eff = np.minimum(np.divide(samp_deps, denom), samp_deps)
    rbar_const = mean_series_intercorrelation(zero_mean_data, "pearson", min_seg_ratio)
    stabilized_means = np.multiply(mean_rwis, np.sqrt(n_eff * rbar_const))

    if running_rbar:
        stabilized_chron =  pd.DataFrame(data={"Adjusted CRN": stabilized_means + mean_val, "Running rbar": rbar_array, "Sample depth": samp_deps}, index=reg_chron.index)
    else:
        stabilized_chron =  pd.DataFrame(data={"Adjusted CRN": stabilized_means + mean_val, "Sample depth": samp_deps}, index=reg_chron.index)

    print("SUCCESS!\n")
    return stabilized_chron
    
def pad_rbar_array(rbar_array):
    # double check that rbar cannot be 0
    first = 0
    first_valid = 0
    for val in rbar_array:
        if val != 0 and not np.isnan(val):
            first = val
            break
        first_valid += 1

    last = 0
    last_valid = len(rbar_array) - 1
    for val in np.flip(rbar_array):
        if val != 0 and not np.isnan(val):
            last = val
            break
        last_valid -= 1
    
    rbar_array[:first_valid] = np.full(first_valid, first) #should be  np.full(first_valid, 1)
    rbar_array[last_valid:] = np.full(len(rbar_array) - last_valid, last) #should be np.full(len(rbar_array) - last_valid, last)

    return rbar_array