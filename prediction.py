import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import timedelta
from tools.data_tools import *
from model import ARMAModel
from statsmodels.tsa.stattools import adfuller



def ARMA(df):
    df_serie =  to_clean_time_series(df)
    df_diff = df_serie.copy(deep = True)
    df_diff['appts_per_listing'] = df_serie['appts_per_listing'].diff().replace(np.nan, 0)

    armaModel = ARMAModel(df_serie,df_diff,_lags=0)

    prediction = st.number_input('Insert the number of steps in the future you want to visualize',max_value =5,min_value =0,value =0,step =1)
    st.write("*For this model, the maximum recomended steps are 2*")

    
    window = 48
    if(len(df) < 60):
        window = int(len(df)*0.6)

    x_dates = df['start_date']
    if(prediction > 0):
        lastdate = df.iloc[[-1]]['start_date'].values.ravel()[0]
        last_dates = []
        last_dates.append(lastdate) 
        for i in range(1,prediction+1):
            lastdate = to_start_date(lastdate + pd.to_timedelta(20,unit="D"))
            last_dates.append(lastdate)
        prediction_dates = last_dates[1:]
        x_dates = pd.concat([df['start_date'],pd.Series(prediction_dates)],ignore_index=True)





    df_window = df.tail(window-prediction)
    df_diff_window = df_diff.tail(window-prediction)
    x_dates_window = x_dates.tail(window)
    
    st.subheader("Historical Plot")
    stacionality_test = adfuller(df['appts_per_listing'].ravel())
    predict_serie = armaModel.predict(prediction)
    predict_serie_window = predict_serie[len(predict_serie)-window:]

    st.write("*Test of Stacionality :* ",stacionality_test[1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_dates_window, y=df_window['appts_per_listing'],
                                     mode='lines+markers',
                                     name="Historical", connectgaps=True))
    
    if(stacionality_test[1] <= 0.05):
        fig.add_trace(go.Scatter(x=x_dates_window, y=predict_serie_window.values,
                                        mode='lines+markers',
                                        name="Prediction", connectgaps=True))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.warning("This data cannot be forecasted")


    


    st.subheader("Historical Difference Plot")
    stacionality_test_diff = adfuller(df_diff['appts_per_listing'].ravel())
    predict_serie_diff = armaModel.predict_diff(prediction)
    predict_serie_diff_window = predict_serie_diff[len(predict_serie_diff)-window:]
    st.write("*Test of Stacionality :* ",stacionality_test_diff[1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_dates_window, y=df_diff_window['appts_per_listing'],
                                     mode='lines+markers',
                                     name="Historical", connectgaps=True))
    
    if(stacionality_test_diff[1] <= 0.05):
        fig.add_trace(go.Scatter(x=x_dates_window, y=predict_serie_diff_window.values,
                                    mode='lines+markers',
                                    name="Prediction", connectgaps=True))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.warning("This data cannot be forecasted")


    


def page_prediction():
    df = load_data()
    st.title("Prediction")
    st.header("Parameters Selection")
    zip_code_option = st.selectbox(
        'Select a zip code: ',
        list(set(df['zip_code'].values)))

    df_filtered_raw = df[df['zip_code'] == zip_code_option].sort_values(by='start_date').reset_index(drop=True)
    df_filtered_raw['start_date'] = pd.to_datetime(df['start_date'],format = '%Y-%m-%d %H:%M:%S')
    st.write(df_filtered_raw)
    ARMA(df_filtered_raw)

