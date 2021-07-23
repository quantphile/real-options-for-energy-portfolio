import pandas as pd

def read_data(file_name):
    df = pd.read_csv(file_name)
    return df

def set_index_as_date(df):
    df.set_index('DATE', inplace=True)
    return df

def split_data(df):
    df_dict = {}
    for period in df['PERIOD'].unique():
        df_dict[period] = df.loc[df['PERIOD'] == period]
    return df_dict