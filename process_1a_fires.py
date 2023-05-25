import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from datetime import datetime
import os


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
    
'''
Transform the  date column into cyclical sin/cos values based on the day of the year
'''
def convert_date_to_cyclical(df, date_column):
    
    df[date_column] = pd.to_datetime(df[date_column])
    df['day_of_year'] = df[date_column].apply(lambda x: x.timetuple().tm_yday)

    df["DOY_SIN"] = sin_transformer(365).fit_transform(df["day_of_year"])
    df["DOY_COS"] = cos_transformer(365).fit_transform(df["day_of_year"])
    
    df = df.drop('day_of_year', axis=1)
    return df
    
def process_fires(df):
    # Drop rows if missing information
    df = df.rename({'HubDist': 'HWY_DSTNC'}, axis = 1)
    required_cols = ['IGNITION', 'POO_LATITUDE', 'POO_LONGITUDE', 'HWY_DSTNC', 'STATISTICAL_CAUSE']
    df = df.dropna(subset=required_cols, how = 'any')    

    # Convert to datetime.date variable type and fix two errors
    df['IGNITION'] = df.IGNITION.apply(lambda x: x.date())
    df.loc[df['IGNITION'] == datetime(205, 8, 5).date(), "IGNITION"] = datetime(2005, 8, 5).date()
    df.loc[df['IGNITION'] == datetime(1010, 9, 21).date(), "IGNITION"] = datetime(2010, 9, 21).date()
    
    # Add sin/cos values for the date
    df = convert_date_to_cyclical(df, 'IGNITION')
    
    # Only keep needed columns
    keep_cols = ['OBJECTID', 'FIRE_NAME', 'POO_LATITUDE', 'POO_LONGITUDE', 'IGNITION',
                 'STATISTICAL_CAUSE', 'TOTAL_ACRES_BURNED', 'HubName', 'HWY_DSTNC', 'DOY_SIN', 'DOY_COS']
    df = df[keep_cols]
    return df


if __name__ == '__main__':
    # Read in the required datasets and preform preprocessing
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data')
    df = pd.read_excel(r'FIRESTAT_YRLY_2005-2015_distance.xlsx')
    
    df = process_fires(df)
    
    # Save processed df
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data\processed_data')
    today = datetime.today().strftime('%Y%m%d')
    df.to_pickle(f'processed_fires_{today}.pkl')

