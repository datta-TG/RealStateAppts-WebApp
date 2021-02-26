import datetime
import streamlit as st
import pandas as pd
from tools.data_tools import *


@st.cache(allow_output_mutation=True)
def get_removed():
    removing_dates = {'start_date':[],'end_date':[]}
    return removing_dates

def removing_dates(df):
    st.subheader("Removing data by period of time")
    dates = df['start_date'].unique()
    min_date = datetime.datetime.strptime(pd.to_datetime(
        str(min(dates))).strftime('%Y-%m-%d'), '%Y-%m-%d').date()
    max_date = datetime.datetime.strptime(pd.to_datetime(
        str(max(dates))).strftime('%Y-%m-%d'), '%Y-%m-%d').date()

    date = st.date_input("Select the dates you don't want to consider",
                         min_date, min_value=min_date, max_value=max_date)
    date_start = to_start_date(date)
    date_end = to_end_date(date)
    st.write("The period selected is:")
    st.write("Start date :", date_start)
    st.write("End date :", date_end)

    removed_header = st.empty()
    removed_list = st.empty()

    button1, button2, button3 = st.beta_columns([.2,.2,1])
    add_button = button1.button("Add")
    clear_button = button2.button("Clear")
    remove_button = button3.button("Remove!")

    if(add_button):
        if((date_start not in get_removed()['start_date']) and (str(date_start) in dates)):
            get_removed()['start_date'].append(date_start)
            get_removed()['end_date'].append(date_end)

    if(clear_button):
        get_removed()['start_date'] = []
        get_removed()['end_date'] = []
    
    if(remove_button):
        for start_date in get_removed()['start_date']:
            df.drop(df[df['start_date']== str(start_date)].index, inplace=True)
        save_database(df)
        get_removed()['start_date'] = []
        get_removed()['end_date'] = []
        st.success("Periods removed! Please refresh above")
    

    if(len(get_removed()['start_date']) > 0):
        removed_header.subheader("Dates to be removed")
        removed_list.dataframe(pd.DataFrame(get_removed()).sort_values(by="start_date"))
    

def page_database_config():
    df = load_data()
    
    st.title("Database Configuration")
    st.header("Database Summary")
    st.write("Number of registers:",len(df))
    table,_, refresh = st.beta_columns([3,.2,1])
    table.dataframe(df)
    refresh.button("Refresh")
    with st.beta_expander("Remove data"):
        removing_dates(df)
