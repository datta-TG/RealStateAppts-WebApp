import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta
from model import LGBMRegressorModel
from tools.data_tools import *





def page_prediction():
    st.error("NOT WORKING")
    df = load_data()
    st.title("Prediction")
    st.header("Parameters Selection")
    zip_code_option = st.selectbox(
        'Select a zip code: ',
        list(set(df['zip_code'].values)))

    df_filtered_raw = df[df['zip_code'] == zip_code_option].sort_values(by='start_date')
    last_date_row = df_filtered_raw.iloc[-1]
    last_date = datetime.datetime.strptime(last_date_row['start_date'], '%Y-%m-%d').date()



    date_option = st.date_input(
        "Select a start date, remember predictions are only done with first or last two weeks of each month", datetime.date.today(), max_value = datetime.date.today() + timedelta(days = 365),min_value= to_start_date(last_date + timedelta(days = 16)))
    start_date_option = to_start_date(date_option)
    end_date_option = to_end_date(date_option)
    st.write("Your *start date* is:",start_date_option)
    st.write("Your *end date* is:",end_date_option)

    listings_option = st.number_input('Select the number of listings:',min_value=0,value= 10,step = 1)
    button = st.button("Make Prediction")
    if(button):
        st.header('Prediction')
        model = LGBMRegressorModel(load = True)
        prediction_data = {'zip_code':[str(zip_code_option)],'start_date':[start_date_option],'listings':[listings_option]}
        prediction_data = pd.DataFrame.from_dict(prediction_data)
        result = model.predict(prediction_data)
        result['Label'] = result['Label'].apply(round)


        result.rename(columns={'Label':'appts'},inplace= True)
        st.write(result)

        
        df_filtered = to_clean_time_series(df_filtered_raw)

        result.loc[-1] = last_date_row
        df_filtered_prediction = to_clean_time_series(result)
        

        st.subheader('Line plot')
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_filtered_prediction.index,
                                        y=df_filtered_prediction.appts, mode='lines+markers',name = 'prediction',line = dict(color='firebrick')))

        fig.add_trace(go.Scatter(x=df_filtered.index,
                                        y=df_filtered.appts, mode='lines+markers',name = 'data',line = dict(color='royalblue')))

        
        st.plotly_chart(fig, use_container_width=True)

