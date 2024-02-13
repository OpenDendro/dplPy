import dplpy as dpl

def test_detrend_all_fits_residual():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    spline_data = dpl.detrend(data, fit="spline", method="residual", plot=False)
    modnegex_data = dpl.detrend(data, fit="ModNegEx", method="residual", plot=False)
    hugershoff_data = dpl.detrend(data, fit="Hugershoff", method="residual", plot=False)
    linear_data = dpl.detrend(data, fit="linear", method="residual", plot=False)
    horizontal_data = dpl.detrend(data, fit="horizontal", method="residual", plot=False)

    # TODO: assert detrended data for correctness

def test_detrend_all_fits_difference():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    spline_data = dpl.detrend(data, fit="spline", method="difference", plot=False)
    modnegex_data = dpl.detrend(data, fit="ModNegEx", method="difference", plot=False)
    hugershoff_data = dpl.detrend(data, fit="Hugershoff", method="difference", plot=False)
    linear_data = dpl.detrend(data, fit="linear", method="difference", plot=False)
    horizontal_data = dpl.detrend(data, fit="horizontal", method="difference", plot=False)

    # TODO: assert detrended data for correctness

# Commented out because plots block execution in vscode. WIP
# def test_detrend_all_fits_plot():
#     data = dpl.readers("./integs/data/csv/ca533.csv")

#     spline_data = dpl.detrend(data, fit="spline", method="difference", plot=True)
#     modnegex_data = dpl.detrend(data, fit="ModNegEx", method="difference", plot=True)
#     hugershoff_data = dpl.detrend(data, fit="Hugershoff", method="difference", plot=True)
#     linear_data = dpl.detrend(data, fit="linear", method="difference", plot=True)
#     horizontal_data = dpl.detrend(data, fit="horizontal", method="difference", plot=True)

#     # TODO: assert detrended data for correctness
