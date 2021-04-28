from tools.data_tools import *
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
from datetime import timedelta

class ARMAModel:
    valores_ar = range(2,7)
    valores_ma = range(2,7)
    model = None
    model_diff = None
    y = None
    y_diff = None

    def __init__(self,df_serie,df_diff,_lags = 0):

        self.y = df_serie['appts_per_listing']
        self.y_diff = df_diff['appts_per_listing']
        if(_lags == 0):
            BIC = []
            models = []
            BIC_diff = []
            models_diff = []
            for ar in self.valores_ar:
                for ma in self.valores_ma:
                    model = ARIMA(self.y, order=(ar, 0, ma))
                    model_diff = ARIMA(self.y_diff, order=(ar, 0, ma))
                    model_fit = model.fit()
                    model_diff_fit = model_diff.fit()
                    models.append(model_fit)
                    models_diff.append(model_diff_fit)
                    BIC.append(model_fit.bic)
                    BIC_diff.append(model_diff_fit.bic)
            index = BIC.index(min(BIC))
            index_diff = BIC_diff.index(min(BIC_diff))
            self.model = models[index]
            self.model_diff = models_diff[index_diff]
        else:
            model = ARIMA(self.y, order=(_lags, 0, _lags))
            model_diff = ARIMA(self.y_diff, order=(_lags, 0, _lags))
            self.model = model.fit()
            self.model_diff = model_diff.fit()

    def predict(self,prediction=0):
        return self.model.predict(0,len(self.y_diff)-1+prediction)

    def predict_diff(self,prediction=2):
        return self.model_diff.predict(0,len(self.y_diff)-1+prediction)

