import os
import datetime
import streamlit as st
import pandas as pd
from tools.data_tools import *


@st.cache
def upload_data(database):
    with st.spinner(text='Uploading...'):
        df_uploaded = pd.read_excel(database)
        
    return df_uploaded

@st.cache
def validation(df_raw):
    try:
        df = df_raw[['start_date', 'zip_code', 'appts_per_listing','appts']]
        return True
    except KeyError:
        return False



@st.cache
def cleaning_data(df):
    zipcodelen = 5
    df_out = df.copy(deep=True)
    df_out['zip_code'] = df_out['zip_code'].apply(
        lambda x: str(x)[:zipcodelen])
    df_out = quitOutliers(df_out, 'appts_per_listing')
    df_out['listings'] = df_out['appts']/df_out['appts_per_listing']

    years = get_years(df_out)
    enough_registers = (max(years)-min(years) + 1) * 24
    df_out = df_out.groupby('zip_code').filter(lambda x: len(x) >= enough_registers)
    df_out.dropna(inplace=True)
    df_out = df_out.groupby(['zip_code', 'start_date']).agg(
        {'listings': 'sum','appts':'sum'}).reset_index(level=['start_date', 'zip_code'])
    df_out['appts_per_listing'] = df_out['appts']/df_out['listings']
    df_out.drop(['appts','listings'],axis = 1, inplace = True)

    return df_out

def page_upload_database():
    st.title("Upload new database")
    st.write("The new database **must** have this 4 columns:**appts**, **appts_per_listing** , **start_date** and **zip_code**")
    database = st.file_uploader("Upload the new database", type=['xlsx','xls'])
    if(database):
        error_message = st.empty()
        df_raw = upload_data(database)
        if(validation(df_raw)):
            df = cleaning_data(df_raw)
            st.subheader("New database")
            st.write("Number of registers:",len(df))
            st.dataframe(df)
            st.warning("The actual database will be overrided by this new database, this action cannot be undone, please be sure")
            override_button = st.button("Override!")
            if(override_button):
                save_database(df)
                st.success("The database was replaced sucessfully, you can see the changes in the *database config section*")
        else:
            error_message.write("The database uploaded doesn´t have the right columns.")
            