import pandas as pd
import datetime
import numpy as np

def load_data():
    return pd.read_csv('src/database.csv')

def save_database(df):
    df.to_csv('src/database.csv',index = False)

def to_clean_time_series(df):
    datetime_index = pd.DatetimeIndex(pd.to_datetime(df['start_date']).values)
    return df.set_index(datetime_index).sort_index().drop(['zip_code', 'start_date'], axis=1)


def quitOutliers(df, column):
    mean = df[column].mean()
    std = df[column].std()
    high_limit = mean + 3*std
    low_limit = mean - 3*std
    outliers = df[(df[column] >= high_limit) | (df[column] <= low_limit)]
    df_out = df.drop(outliers.index)
    df_out.reset_index(inplace=True, drop=True)
    return df_out


def to_start_date(date):
    month = date.month
    year = date.year
    inflection_point = datetime.date(year, month, 16)
    if(date < inflection_point):
        return datetime.date(year, month, 1)
    else:
        return inflection_point



def to_end_date(date):
    month = date.month
    year = date.year
    inflection_point = datetime.date(year, month, 16)
    if(date < inflection_point):
        return datetime.date(year, month, 15)
    else:
        next_month = date.replace(day=28) + datetime.timedelta(days=4)
        return next_month - datetime.timedelta(days=next_month.day)

def get_years(df):
    return list(
        set(np.array(pd.DatetimeIndex(df['start_date']).year)))