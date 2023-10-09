from rbar import get_running_rbar, mean_series_intercorrelation
from chron import chron
import numpy as np
import pandas as pd


def chron_stabilized(rwi_data, win_length=50, biweight=True, running_rbar=False):
    # Add checks and warnings here. window length should be an integer > the chronology
    # length. Window length is also recommended to be >= 30 and < 50% of chronology length.

    # give rbar function a range of years (window length) to calculate rbar for
    # calculate rbar for that window, using either osborn's or frank's or 67spline
    # get rbar for each relevant segment of the dataframe

    num_years = rwi_data.shape[0]

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
        rbar_array[target_index] = get_running_rbar(data_segment)

    rbar_array = pad_rbar_array(rbar_array)

    reg_chron = chron(zero_mean_data, biweight=biweight, plot=False)

    mean_rwis = reg_chron["Mean RWI"].to_numpy()
    samp_deps = reg_chron["Sample depth"].to_numpy()
    denom = np.multiply(samp_deps-1, rbar_array) + 1

    n_eff = np.divide(samp_deps, denom), rbar_array

    rbar_const = mean_series_intercorrelation(zero_mean_data, "Pearson", 0)

    stabilized_means = np.multiply(mean_rwis, np.sqrt(n_eff * rbar_const))

    if running_rbar:
        stabilized_chron =  pd.DataFrame(data={"Adjusted CRN": stabilized_means + mean_val, "Running rbar": rbar_array, "Sample depth": samp_deps}, index=reg_chron.index)
    else:
        stabilized_chron =  pd.DataFrame(data={"Adjusted CRN": stabilized_means + mean_val, "Sample depth": samp_deps}, index=reg_chron.index)

    print(stabilized_chron.to_string())

    return stabilized_chron
    
def pad_rbar_array(rbar_array):
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