from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, graphics, pacf

def autoreg(data):
    AR_data = AutoReg(data, 1, old_names=False)
    model = AR_data.fit()
    print(model.params)
    print(model.summary())