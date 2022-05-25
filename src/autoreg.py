from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, graphics, pacf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def autoreg(data):
    #AR_data = ar_select_order(data, 20, ic='aic', old_names=False)
   
    #results = AR_data.model.fit()
    
    results = AutoReg(data, 20).fit()

    print(results.summary())
    pars = results.params


    fig = plt.figure(figsize=(16, 9))
    fig = results.plot_diagnostics(fig=fig, lags=30)
    
    
    x = data.index.to_numpy()
    y = data.to_numpy()
    print(y)
    print()
    yi = predict_values(y, pars)
    
    print(yi)
    print()

    res = y[len(pars)-1:] - yi
    print(res)
    mean = np.mean(y)

    for i in range(len(res)):
        res[i] += mean

    print(res)

    plt.clf()
    plt.plot(x[len(pars)-1:], res, "-")
    plt.show()

    
def autoreg_array(x, y):
    AR_data = ar_select_order(y, 5, ic='aic', old_names=False)
    results = AR_data.model.fit()

    print(results.summary())
    fig = plt.figure(figsize=(16, 9))
    fig = results.plot_diagnostics(fig=fig, lags=30)
    
    plt.clf()
    print()
    yi = results.get_prediction().df
    print(results.get_prediction().predicted)
    print(results.get_prediction().var_resid)
    print(yi)
    
    print("\nPrinting residuals now\n")
    res = y/yi
    print(res)



def predict_values(data_array, params):
    mean = np.mean(data_array)
    results = []

    for i in range((len(params)-1), len(data_array)):
        pred = params[0]
        for j in range(1, len(params)):
            pred += (params[j] * data_array[i-j])
        results.append(pred)
    return np.asarray(results)