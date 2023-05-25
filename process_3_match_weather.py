import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.neighbors import KDTree

    
def locate_fire_weather_var_kdtree(df, weather_df, col):
    print(f'\nBuilding KDTrees, {datetime.now().time()}')
    var_df = weather_df[weather_df[col].notna()]
    
    df[col] = df.apply(lambda x: build_tree_find_closest(col, x.IGNITION, np.array([x.POO_LATITUDE, x.POO_LONGITUDE]), 
                                                         var_df), axis = 1)
    print(f'Completed matching, {datetime.now().time()}')
    return df

def build_tree_find_closest(col, day, location, var_df):
    tree_df = var_df[(var_df[col].notna()) & (var_df.DATE == day)]
    tree = KDTree(tree_df[['LATITUDE', 'LONGITUDE']].to_numpy())
    dist, ind = tree.query(location.reshape(1, -1), k=1) #find the closest weather station
    return tree_df[col].iloc[ind[0][0]]


if __name__ == '__main__':
    # Read in the required datasets
    today = datetime.today().strftime('%Y%m%d')
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data')
    stations_df = pd.read_pickle(r'California Weather Stations\CA_weather_stations.pkl')
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data\processed_data')
    weather_df = pd.read_pickle(f'processed_weather_{today}.pkl') #r'processed_weather_20230227.pkl'
    weather_df = pd.merge(weather_df, stations_df[['STATION', 'LATITUDE', 'LONGITUDE']], how = 'inner', on = 'STATION')  
    model_df = pd.read_excel(f'fire_modeling_{today}.xlsx')    #'fire_modeling_20230222.xlsx'
    
    
    # Add in the weather variables (these take 5-60 min to run each)
    model_df = locate_fire_weather_var_kdtree(model_df, weather_df, 'EVAP')
    model_df = locate_fire_weather_var_kdtree(model_df, weather_df, 'AWND')
    model_df = locate_fire_weather_var_kdtree(model_df, weather_df, 'TMAX')
    model_df = locate_fire_weather_var_kdtree(model_df, weather_df, 'TMIN')
    model_df = locate_fire_weather_var_kdtree(model_df, weather_df, 'PRCP_ROLLING')
        

    # Repeat for the random points
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data\processed_data')
    rand_df = pd.read_excel(f'random_points_{today}.xlsx') #r'random_points_20230227.xlsx'
    
    rand_df.columns = ['IGNITION', 'POO_LATITUDE', 'POO_LONGITUDE', 'region_id']
    rand_df = locate_fire_weather_var_kdtree(rand_df, weather_df, 'EVAP')
    rand_df = locate_fire_weather_var_kdtree(rand_df, weather_df, 'AWND')
    rand_df = locate_fire_weather_var_kdtree(rand_df, weather_df, 'TMAX')
    rand_df = locate_fire_weather_var_kdtree(rand_df, weather_df, 'TMIN')
    rand_df = locate_fire_weather_var_kdtree(rand_df, weather_df, 'PRCP_ROLLING')
    rand_df.columns = ['date', 'latitude', 'longitude', 'region_id', 'EVAP',
                       'AWND', 'TMAX', 'TMIN', 'PRCP_ROLLING']
    
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data\processed_data')
    model_df.to_excel(f'fire_modeling_{today}.xlsx', index = False)
    rand_df.to_excel(f'random_points_weather_{today}.xlsx', index = False)
