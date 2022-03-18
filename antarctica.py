import numpy as np
from sklearn.cluster import DBSCAN
from functools import reduce
import pandas as pd
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import fnmatch
import xarray as xr
import pywt
%matplotlib qt

# %%
base_dir = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Antarctica/'
dir_mavic = base_dir + r'mavic_Matt\mavic_profiles_gridded/'
file_path = [x for x in glob.glob(dir_mavic + '*.csv') if 'cal_flight' not in x]
file_names = [os.path.basename(x) for x in glob.glob(dir_mavic + '*.csv') if 'cal_flight' not in x]

dir_wind = base_dir + r'Wind_estimates/'
file_path_wind = [x for x in glob.glob(dir_wind + '*.csv')]
file_names_wind = [os.path.basename(x) for x in glob.glob(dir_wind + '*.csv')]

dir_save = base_dir + r'Viet\merged_wind/'

# %%
appended_df_wind = []
for x, x_name in zip(file_path_wind, file_names_wind):
    df_temp = pd.read_csv(x)
    date_ = x_name.split('_')[1]
    time_ = x_name.split('_')[2].split('.')[0].replace('-', ':')
    date_time_ = pd.to_datetime(date_ + ' ' + time_)
    df_temp['datetime'] = date_time_ + pd.to_timedelta(df_temp['Flight time'])
    df_temp = df_temp.set_index('datetime').resample('1s').mean().reset_index()
    appended_df_wind.append(df_temp)

df_wind = pd.concat(appended_df_wind, ignore_index=True)

# %%
for x, x_name in zip(file_path, file_names):
    df = pd.read_csv(x)
    df.dropna(how='all', inplace=True)
    df = df.rename({'date_time': 'datetime'}, axis='columns')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_full = df.merge(df_wind, on='datetime', how='inner')
    file_name_ = '_'.join(x_name.split('_')[3:])
    if df_full.size == 0:
        continue
    else:
        df_full.to_csv(dir_save + 'merged_wind' + file_name_, index=False)
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(df_full.datetime, df_full.Altitude, '.')
        ax[0].set_ylabel('Altitude from wind')
        ax[1].plot(df_full.datetime, df_full.press_bme, '.')
        ax[1].set_ylabel('press_bme')
        for ax_ in ax.flatten():
            ax_.grid()
        fig.savefig(dir_save + 'plots/' + file_name_[:-4] + '.png', bbox_inches='tight')
        plt.close()

# %%

ref = pd.read_csv(base_dir + r'mavic_Matt\weather.csv')
