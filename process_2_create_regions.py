import pandas as pd
import numpy as np
import os
from string import ascii_lowercase
from datetime import datetime
import seaborn as sns
from math import radians, cos, sin, asin, sqrt


'''
Calculate the great circle distance between two points 
on the earth (specified in decimal degrees)
'''
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    ml = km/1.609
    return ml

'''
Divides an area into a specified number of regions, given min/max coordinates 
and returns a dataframe with the bounds for latitude and longitude and assigned
ids for longitude and latitude seperately
Note: the resulting df does not contain every combination of lat/long bounds 
'''
def divide_region(num_lat_boxes, num_long_boxes,
                  min_lat, max_lat, min_long, max_long):
       
    lat_list = np.linspace(min_lat, max_lat, num_lat_boxes + 1)
    long_list = np.linspace(min_long, max_long, num_long_boxes + 1)
    
    regions_df = pd.DataFrame([lat_list[:-1], lat_list[1:],
                               long_list[:-1], long_list[1:]]).T
    regions_df.columns = ['lower_lat', 'upper_lat',
                          'lower_long', 'upper_long']
    
    regions_df['lat_id'] = list(ascii_lowercase[:len(regions_df)])
    regions_df['long_id'] = regions_df.index
    
    lat_len = haversine(0, regions_df.lower_lat.iloc[0], 0, regions_df.upper_lat.iloc[0])
    long_len = haversine(regions_df.lower_long.iloc[0], 0, regions_df.upper_long.iloc[0], 0)
    print(f"\n\nEach bin is {round(lat_len)} in latitude, and {round(long_len)} in longitude\n")
    return regions_df

'''
Assigns each latitude and longitude the correct region that contains the point,
given a region df such as above
'''
def assign_regions(df, lat_col, long_col, regions_df):
    lat_bins = np.insert(regions_df['upper_lat'].values, 0, regions_df['lower_lat'].iat[0])
    long_bins = np.insert(regions_df['upper_long'].values, 0, regions_df['lower_long'].iat[0])
    
    df['lat_range'] = pd.cut(df[lat_col], bins=lat_bins, right=False)
    df['long_range'] = pd.cut(df[long_col], bins=long_bins, right=False)
  
    df['lower_lat'] = df['lat_range'].apply(lambda x: x.left)
    df['upper_lat'] = df['lat_range'].apply(lambda x: x.right)
    df['lower_long'] = df['long_range'].apply(lambda x: x.left)
    df['upper_long'] = df['long_range'].apply(lambda x: x.right)
    
    df = pd.merge(df, regions_df[['lower_lat', 'upper_lat', 'lat_id']].round(3), on = ['lower_lat', 'upper_lat'], how = 'left')
    df = pd.merge(df, regions_df[['lower_long', 'upper_long', 'long_id']].round(3), on = ['lower_long', 'upper_long'], how = 'left')

    df['region_id'] = df['lat_id'] + df['long_id'].astype(str)
    df = df.drop(['lat_range', 'long_range', 'lower_lat', 'upper_lat', 'lat_id', 'lower_long', 'upper_long', 'long_id'], axis = 1)
    return df

def visualize_region_values(regions_df, df, value_col):
    viz_df = pd.DataFrame([a+str(b) for a in regions_df.lat_id for b in regions_df.long_id], 
                          columns = ['region_id'])
    viz_df = pd.merge(viz_df, df[['region_id', value_col]].drop_duplicates(), 
                      how = 'left', on = 'region_id')
    viz_df['latitude_bin'] = viz_df.region_id.apply(lambda x: x[:1])
    viz_df['longitude_bin'] = viz_df.region_id.apply(lambda x: f"{int(x[1:]):02}")
    sns.heatmap(viz_df.pivot(index='latitude_bin', columns='longitude_bin', 
                             values=value_col).sort_index(ascending=False)).set_title(f'{value_col} of Created Regions')
       
'''
Counts the number of fires that occured during the defined years
Returns a min/max normalization of the number of fires as a background rate
NOTE Min/Max normalize DOES NOT subtract min from top (to avoid 0s that aren't really 0)
'''                                                       
def determine_background_rate(df, years):
    if 'BKGD_RT' in df.columns:
        df = df.drop('BKGD_RT', axis = 1)
    # Count the number of fires that occur in each area for defined range of years
    background = df[df.IGNITION.apply(lambda x: x.year).isin(years)]
    background = background.value_counts('region_id').to_frame().reset_index()
    background.columns = ['region_id', 'BKGD_RT']
    
    # Preform min/max normalization by: x/(x.max - x.min)
    background.BKGD_RT = background.BKGD_RT/(background.BKGD_RT.max() - background.BKGD_RT.min())

    # Join back into the dataframe    
    df = pd.merge(df, background, on = ['region_id'])    
    return df

def random_dates(start, end, n=10):
    start_u = start.value//(24*60*60*10**9)
    end_u = end.value//(24*60*60*10**9)
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='D')

def draw_rand_region_pts(regions_df, fire_regions, yrs, n):
    start = pd.to_datetime(f'{yrs[0]}-01-01')
    end = pd.to_datetime(f'{yrs[-1]}-01-01')
       
    rand_dfs = []
    for region in fire_regions:
       lat = regions_df[regions_df.lat_id == region[0]]
       rand_lats = list(np.random.randint(low = lat.lower_lat*10e5, high=lat.upper_lat*10e5, size=n))
       rand_lats = [x / 10e5 for x in rand_lats]
       
       long = regions_df[regions_df.long_id == int(region[1])]
       rand_longs = list(np.random.randint(low = long.lower_long*10e5, high=long.upper_long*10e5, size=n))
       rand_longs = [x / 10e5 for x in rand_longs]
       
       rand_dates = random_dates(start, end, n=n)
       
       rg_df = pd.DataFrame({'date':rand_dates,'latitude': rand_lats, 'longitude': rand_longs, 'region_id':region})
       rand_dfs.append(rg_df)
       
    return pd.concat(rand_dfs)
    
   
if __name__ == '__main__':
    # Read in the required datasets
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data\processed_data')
    today = datetime.today().strftime('%Y%m%d')
    df_old = pd.read_pickle(f'processed_fires_{today}.pkl') #r'processed_fires_20230125.pkl'
    df = pd.read_excel(f"fire_modeling_{today}.xlsx") #"fire_modeling_20230217.xlsx"
    df = pd.concat([df, df_old[~df_old.OBJECTID.isin(df.OBJECTID)]])
    n_bins = 10 # The number of bins to divide california longitude and latitude into
    
    # Create regions
    min_lat, max_lat = 32.534156, 42.009518
    min_long, max_long = -124.409591, -114.131211
    regions_df = divide_region(n_bins, n_bins, min_lat, max_lat, min_long, max_long)
    
    # Assign fires to the region they were located in
    df = assign_regions(df, 'POO_LATITUDE', 'POO_LONGITUDE', regions_df)
    
    # Visualize the number of fires in each region of the full dataset
    viz_df = df.region_id.value_counts().reset_index()
    viz_df.columns = ['region_id', 'Fire Count']
    visualize_region_values(regions_df, viz_df, 'Fire Count')    
    
    # Visualize regions with one fire - these won't be modeled
    viz_df2 = viz_df[viz_df['Fire Count'] == 1].rename(columns = {'Fire Count': 'Single Fire Regions'})
    visualize_region_values(regions_df, viz_df2, 'Single Fire Regions')    
    
    # Add in the background rate for each region
    bkgrd_yrs = list(range(2005, 2008))
    train_test_yrs = list(range(2008, 2016))
    df = determine_background_rate(df, bkgrd_yrs) 
    visualize_region_values(regions_df, df, 'BKGD_RT')
    
    bkgrd_df = df[df.IGNITION.apply(lambda x: x.year).isin(bkgrd_yrs)].reset_index(drop=True)
    model_df = df[df.IGNITION.apply(lambda x: x.year).isin(train_test_yrs)].reset_index(drop=True) #drop years used for background rate
    missing_bkgrd = [x for x in model_df.region_id.unique() if x not in bkgrd_df.region_id.unique()]
    if len(missing_bkgrd) != 0:
        print(f'\n\nThese regions missing background rates: \n{missing_bkgrd}\n')
    
    # Specify fires caused by lightening
    model_df['LGHTNG'] = model_df.STATISTICAL_CAUSE == '1 -  Lightning'
    model_df['DISCOVER_YEAR'] = model_df.IGNITION.apply(lambda x: x.year)
    
    # Create randome day/location dataset to compare against
    fire_regions = model_df.region_id.unique()
    
    n_pts = 100
    rand_pts = draw_rand_region_pts(regions_df, fire_regions, list(range(2008, 2013)), n_pts)
    rand_pts = pd.merge(rand_pts, model_df[['region_id', 'BKGD_RT']].drop_duplicates(), 
                        how = 'left', on = 'region_id')
    
        
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data\processed_data')
    model_df.to_excel(f'fire_modeling_{today}.xlsx', index = False)
    rand_pts.to_excel(f'random_points_{today}.xlsx', index = False)

    
    