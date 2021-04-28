import streamlit as st
from timeseries import page_timeseries
from prediction import page_prediction
from database_config import page_database_config
from upload_database import page_upload_database


def config_sub_menu():
    configPages = {
        "Database Configuration": page_database_config,
        "Upload new database": page_upload_database
    }

    st.sidebar.subheader("Configuration Submenu")
    page = st.sidebar.radio("Select your page", tuple(configPages.keys()))
    # Display the selected page
    configPages[page]()

def main():
    principalPages = {
        "Time Series": page_timeseries,
        "Prediction": page_prediction,
        "Configuration":config_sub_menu
    }

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select your page", tuple(principalPages.keys()))



    # Display the selected page
    principalPages[page]()


if __name__ == "__main__":
    main()
