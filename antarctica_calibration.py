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
from sklearn.metrics import r2_score
import copy
import matplotlib
%matplotlib qt

##########################################################
# calibration Flight
# %%
#########################################################
base_dir = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Antarctica/'
dir_mavic = base_dir + r'mavic_Matt\mavic_profiles_gridded/'
file_path = [x for x in glob.glob(dir_mavic + '*.csv') if 'cal_flight' in x]
file_names = [os.path.basename(x) for x in glob.glob(dir_mavic + '*.csv') if 'cal_flight' in x]

dir_wind = base_dir + r'/Calibration_flights/Wind_estimates/'
file_path_wind = [x for x in glob.glob(dir_wind + '*.csv')]
file_names_wind = [os.path.basename(x) for x in glob.glob(dir_wind + '*.csv')]

dir_save = base_dir + r'Viet\merged_wind/'

appended_df_wind = []
for x, x_name in zip(file_path_wind, file_names_wind):
    df_temp = pd.read_csv(x)
    date_ = x_name.split('_')[1]
    time_ = x_name.split('_')[2].split('.')[0].replace('-', ':')
    date_time_ = pd.to_datetime(date_ + ' ' + time_)
    df_temp['datetime'] = date_time_ + pd.to_timedelta(df_temp['Flight time'])
    # df_temp = df_temp.set_index('datetime').resample('1s').mean().reset_index()
    appended_df_wind.append(df_temp)

df_wind = pd.concat(appended_df_wind, ignore_index=True)

# %%
weather_full = pd.read_csv(base_dir + r'mavic_Matt\weather.csv')
weather_full = weather_full.rename({'date_time': 'datetime'}, axis='columns')
weather_full.datetime = pd.to_datetime(weather_full.datetime)
weather_full = weather_full.drop(['wdir_10m', 'ws_10m'], axis=1)
weather_full['press_1m_at11m'] = weather_full['press_1m'] + 0.7

# %%
for x, x_name in zip(file_path, file_names):
    df = pd.read_csv(x)
    df.dropna(how='all', inplace=True)
    i_min = np.argmin(df.press_bme)
    df['ascend'] = True
    df.loc[i_min:, 'ascend'] = False
    df = df.rename({'date_time': 'datetime'}, axis='columns')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.reset_index(drop=True)

    df_full = df.merge(df_wind, on='datetime', how='left')
    df_full = df_full.merge(weather_full, on='datetime', how='left')

    df_full[['wdir_3s', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
             'temp_1m', 'dewpoint', 'ws_3s', 'ws_gust',
             'voltage']] = df_full[['wdir_3s', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
                                    'temp_1m', 'dewpoint', 'ws_3s', 'ws_gust',
                                    'voltage']].fillna(method='backfill', limit=60)
    df_full[['Altitude', 'Home Distance',
             'Wind Direction', 'Wind Speed']] = df_full[['Altitude', 'Home Distance',
                                                         'Wind Direction', 'Wind Speed']].fillna(method='ffill', limit=10)

    fig, ax = plt.subplots(4, 2, sharex=True, figsize=(16, 9))

    ax[0, 0].plot(df_full.datetime, df_full.press_bme, '.', label='press_bme')
    ax[0, 0].plot(df_full.datetime, df_full.press_1m, '.', label='press_1m_tower')
    ax[0, 0].set_ylabel('Pressure [hPa]')

    ax[0, 1].plot(df_full.datetime, df_full.Altitude, '.', label='Drone Altitude from wind')
    ax[0, 1].set_ylabel('Altitude [m]')

    ax[1, 0].plot(df_full.datetime, df_full.temp_bme, '.', label='temp_bme')
    ax[1, 0].plot(df_full.datetime, df_full.temp_1m, '.', label='temp_1m_tower')
    ax[1, 0].set_ylabel('Temperature [$^0C$]')

    ax[1, 1].plot(df_full.datetime, df_full.rh_bme, '.', label='rh_bme')
    ax[1, 1].plot(df_full.datetime, df_full.rh_1m, '.', label='rh_1m_tower')
    ax[1, 1].set_ylabel('Relative humidity [%]')

    ax[2, 0].plot(df_full.datetime, df_full['temp_sht'], '.', label='temp_sht')
    ax[2, 0].plot(df_full.datetime, df_full['temp_1m'], '.', label='temp_1m_tower')
    ax[2, 0].set_ylabel('Temperature [$^0C$]')

    ax[2, 1].plot(df_full.datetime, df_full['rh_sht'], '.', label='rh_sht')
    ax[2, 1].plot(df_full.datetime, df_full['rh_1m'], '.', label='rh_1m_tower')
    ax[2, 1].set_ylabel('Relative humidity [%]')

    ax[3, 0].plot(df_full.datetime, df_full['Wind Speed'], '.', label='Wind speed')
    ax[3, 0].plot(df_full.datetime, df_full['ws_3s'], '.', label='wind_3m_tower')
    ax[3, 0].set_ylabel('Wind Speed [m/s]')

    ax[3, 1].plot(df_full.datetime, df_full['Wind Direction'], '.', label='Wind direction')
    ax[3, 1].plot(df_full.datetime, df_full['wdir_3s'], '.', label='wind_direction_3m_tower')
    ax[3, 1].set_ylabel('Wind direction [$^0$]')

    for ax_ in ax.flatten():
        ax_.legend()
        ax_.grid()

    fig.savefig(base_dir + r'Viet/cal_flights/' + x_name + '_timeseries.png',
                bbox_inches='tight')

# %%
df = pd.DataFrame({})
for x in file_path:
    df_ = pd.read_csv(x)
    df_.dropna(how='all', inplace=True)
    i_min = np.argmin(df_.press_bme)
    df_['ascend'] = True
    df_.loc[i_min:, 'ascend'] = False
    df = df.append(df_)
df = df.rename({'date_time': 'datetime'}, axis='columns')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.reset_index(drop=True)

df_full = df.merge(df_wind, on='datetime', how='left')
df_full = df_full.merge(weather_full, on='datetime', how='left')

df_full[['wdir_3s', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
         'temp_1m', 'dewpoint', 'ws_3s', 'ws_gust',
         'voltage']] = df_full[['wdir_3s', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
                                'temp_1m', 'dewpoint', 'ws_3s', 'ws_gust',
                                'voltage']].fillna(method='backfill', limit=60)
df_full[['Altitude', 'Home Distance',
         'Wind Direction', 'Wind Speed']] = df_full[['Altitude', 'Home Distance',
                                                     'Wind Direction', 'Wind Speed']].fillna(method='ffill', limit=10)

# %%
data_plot = df_full
fig, ax = plt.subplots(4, 2, figsize=(12, 9))
for ax_, x, y in zip(ax.flatten(),
                     ['press_1m', 'temp_1m', 'rh_1m', 'temp_1m', 'rh_1m',
                      'ws_3s', 'wdir_3s'],
                     ['press_bme', 'temp_bme', 'rh_bme', 'temp_sht', 'rh_sht',
                      'Wind Speed', 'Wind Direction']):

    ax_.plot(data_plot[x], data_plot[y], "+",
             ms=5, mec="k", alpha=0.1)
    ax_.set_xlabel(x)
    ax_.set_ylabel(y)
    ax_.grid()
    line = data_plot[[x, y]].copy()
    line = line.dropna()
    z = np.polyfit(line[x],
                   line[y], 1)

    y_hat = np.poly1d(z)(line[x])
    ax_.plot(line[x], y_hat, "r-", lw=1)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(line[y], y_hat):0.3f}$"
    ax_.text(0.05, 0.95, text, transform=ax_.transAxes,
             fontsize=10, verticalalignment='top')
    axline_min = np.min((np.min(data_plot[x]), np.min(data_plot[y])))
    axline_max = np.min((np.max(data_plot[x]), np.max(data_plot[y])))
    ax_.axline((axline_min, axline_min),
               (axline_max, axline_max),
               color='grey', linewidth=0.5, ls='--')
    ax_.grid()
fig.subplots_adjust(hspace=0.5)
fig.savefig(base_dir + r'Viet/cal_flights/compare.png',
            bbox_inches='tight')
# %%
