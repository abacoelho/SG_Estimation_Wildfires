import pandas as pd
import os
import seaborn as sns


def check_corr(df, col1, col2):
    sub_df = df[df[col1].notna() & df[col2].notna()]
    corr_value = sub_df[col1].corr(sub_df[col2])
    print(f'\n{col1} and {col2} Correlation: \n{corr_value}')
    sns.lmplot(col1, col2, data=sub_df, fit_reg=True)
    return corr_value

if __name__ == '__main__':
    # Read in data and preprocess
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data')
    df = pd.read_pickle(r'California Weather Data\CA Daily Merged.pkl')
    df_evap = pd.read_pickle(r'Evaporation 2008-2014\CA Daily Evaporation.pkl')
    
    df = pd.merge(df, df_evap, on = ['STATION', 'DATE'], how = 'left')
    df['EVAP'] = df['EVAP_x'].fillna(df['EVAP_y'])
    del df_evap    

    df.EVAP = df.EVAP.where(df.EVAP < df.EVAP.quantile(q = 0.9999))
    # check the correlation between precipitation and evap
    col1 = 'EVAP'
    _ = check_corr(df, col1, 'PRCP')
    _ = check_corr(df, col1, 'TMAX')
    _ = check_corr(df, col1, 'AWND') # there are no rows where awnd and evap are both present
    
    df['interaction'] = df.TMAX*df.PRCP
    _ = check_corr(df, col1, 'interaction')
    
    
    # Repeat again with the modeling dataset
    os.chdir(r'C:\Users\acoel\Documents\MS Stats\Research\Wildfire\data\processed_data')
    df = pd.read_excel(r'fire_modeling_20230210.xlsx')
    col1 = 'AREA_EVAP'
    _ = check_corr(df, col1, 'AREA_PRCP_ROLLING')
    _ = check_corr(df, col1, 'AREA_TMAX')
    _ = check_corr(df, col1, 'AREA_AWND') # there are no rows where awnd and evap are both present
    
    df['interaction'] = df.AREA_TMAX*df.AREA_PRCP_ROLLING*df.AREA_AWND
    _ = check_corr(df, col1, 'interaction')
    

