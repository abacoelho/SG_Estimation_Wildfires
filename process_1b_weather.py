import pandas as pd
import numpy as np
from datetime import datetime
import os


def remove_outliers(df):   
    # TMIN and TMAX: removing values outside the 0.001 and .999 percentiles
    df.TMIN = df.TMIN.where(df.TMIN > df.TMIN.quantile(q = 0.001))
    df.TMIN = df.TMIN.where(df.TMIN < df.TMIN.quantile(q = 0.999))
    
    df.TMAX = df.TMAX.where(df.TMAX > df.TMAX.quantile(q = 0.001))
    df.TMAX = df.TMAX.where(df.TMAX < df.TMAX.quantile(q = 0.999))
    
    # AWND: values beyond the .9999 percentile
    df.AWND = df.AWND.where(df.AWND < df.AWND.quantile(q = 0.9999))
    # PRCP: Going to keep all the values
    return df

'''
Remove rows that variables are all NA or an index value is NA
Convert outlier values to NAs
'''
def process_weather(df):
    index_cols = ['STATION', 'DATE']
    weather_cols = ['AWND', 'PRCP', 'EVAP', 'TMAX', 'TMIN']
    
    # Drop the columns that won't be used
    df = df[index_cols + weather_cols]
    
    # Remove rows without date or station
    df = df[df.STATION.notna()]
    df = df[df.DATE.notna()]
 
    # Convert outliers to NA
    df = remove_outliers(df)
    
    # Remove rows where key weather variables are all NA
    df = df[~df[weather_cols].isnull().all(1)]    
    return df
   
'''
Create a 4-day weighted rolling precipitation average
'''
def create_rolling_precip(df):
    # Sort the column into the correct order for calculations
    df = df.sort_values(['STATION', 'DATE'])
    df = df.reset_index(drop = True)
    
    # Create new 4-day rolling weighted average of precipitation for each station
    weights = [0.1, 0.2, 0.3, 0.4]
    df['PRCP_ROLLING'] = df.groupby('STATION')['PRCP'].rolling(4).apply(lambda x: np.sum(weights*x)).reset_index(drop = True)
    return df

'''
Keep only dates on which fires occured
'''
def subset_weather(df, fire_dates):   
    # Convert dates to datetime.date
    df['DATE'] = df.DATE.apply(lambda x: x.date())
    
    # Drop dates that do not have fires
    df = df[df.DATE.isin(fire_dates)]
    return df
 
   
if __name__ == '__main__':
    # Read in data and preprocess
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data')
    df = pd.read_pickle(r'California Weather Data\CA Daily Merged.pkl')
    df_evap = pd.read_pickle(r'Evaporation 2008-2014\CA Daily Evaporation.pkl')
    
    df = pd.merge(df, df_evap, on = ['STATION', 'DATE'], how = 'left')
    df['EVAP'] = df['EVAP_x'].fillna(df['EVAP_y'])
    del df_evap    

    # Gather the dates fires occured (going to drop weather on non-fire days)
    fire_df = pd.read_excel(r'FIRESTAT_YRLY_2005-2015_distance.xlsx')
    fire_dates = fire_df[fire_df.IGNITION.notna()].IGNITION.apply(lambda x: x.date())
    fire_dates = fire_dates.drop_duplicates()
    rand_df = pd.read_excel(r'processed_data\random_points_20230227.xlsx')
    rand_dates = rand_df[rand_df.date.notna()].date.apply(lambda x: x.date()).drop_duplicates()
    del fire_df, rand_df
    dates = list(set(fire_dates.tolist()+rand_dates.tolist()))
    
    df = process_weather(df)
    df = create_rolling_precip(df)
    df = subset_weather(df, dates)
    
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data\processed_data')
    today = datetime.today().strftime('%Y%m%d')
    df.to_pickle(f'processed_weather_{today}.pkl')
