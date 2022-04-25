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

################################
# %%
################################
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
ref = pd.DataFrame({})
weather_full = pd.read_csv(base_dir + r'mavic_Matt\weather.csv')
weather_full = weather_full.rename({'date_time': 'datetime'}, axis='columns')
weather_full.datetime = pd.to_datetime(weather_full.datetime)

# %%

for file in glob.glob(base_dir + r'Viet\merged_wind/*.csv'):
    df_name = file.split('\\')[-1][:-4]
    df = pd.read_csv(file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df[['datetime', 'press_bme']] = df[['datetime', 'press_bme']].set_index(
        'datetime').rolling('10s').median().reset_index()
    weather = weather_full[weather_full.datetime.dt.date == df['datetime'][0].date()]
    weather = weather.set_index('datetime').resample('1S').bfill().reset_index()
    temp = df.merge(weather, on='datetime', how='left')
    temp[['Home Distance', 'Wind Direction',
          'Wind Speed']] = temp[['Home Distance', 'Wind Direction',
                                 'Wind Speed']].fillna(method='backfill', limit=10)
    temp[['wdir_3s', 'wdir_10m', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
          'temp_1m', 'dewpoint', 'ws_3s', 'ws_10m', 'ws_gust',
          'voltage']] = temp[['wdir_3s', 'wdir_10m', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
                              'temp_1m', 'dewpoint', 'ws_3s', 'ws_10m', 'ws_gust',
                              'voltage']].fillna(method='backfill', limit=60)
    i_min = np.argmin(temp.press_bme)
    temp['ascend'] = True
    temp.loc[i_min:, 'ascend'] = False

    intersec = temp[np.abs(temp.press_1m - temp.press_bme) < 0.3]
    intersec = intersec.reset_index(drop=True)
    ref = ref.append(intersec)

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(16, 9))

    ax[0, 0].plot(temp.datetime, temp.press_bme, '.', label='press_bme')
    ax[0, 0].plot(temp.datetime, temp.press_1m, '.', label='press_1m_tower')
    ax[0, 0].set_ylabel('Pressure [hPa]')

    ax[1, 0].plot(temp.datetime, temp.temp_bme, '.', label='temp_bme')
    ax[1, 0].plot(temp.datetime, temp.temp_1m, '.', label='temp_1m_tower')
    ax[1, 0].set_ylabel('Temperature [$^0C$]')

    ax[2, 0].plot(temp.datetime, temp['Wind Speed'], '.', label='Wind speed')
    ax[2, 0].plot(temp.datetime, temp['ws_3s'], '.', label='wind_3m_tower')
    ax[2, 0].set_ylabel('Wind Speed [m/s]')

    ax[0, 1].plot(temp.datetime, temp.Altitude, '.', label='Drone Altitude from wind')
    ax[0, 1].set_ylabel('Altitude [m]')

    ax[1, 1].plot(temp.datetime, temp.rh_bme, '.', label='rh_bme')
    ax[1, 1].plot(temp.datetime, temp.rh_1m, '.', label='rh_1m_tower')
    ax[1, 1].set_ylabel('Relative humidity [%]')

    ax[2, 1].plot(temp.datetime, temp['Wind Direction'], '.', label='Wind direction')
    ax[2, 1].plot(temp.datetime, temp['wdir_3s'], '.', label='wind_direction_3m_tower')
    ax[2, 1].set_ylabel('Wind direction [$^0$]')

    for ax_ in ax.flatten():
        ax_.legend()
        ax_.grid()

    for vline in intersec.datetime:
        for ax_ in ax.flatten():
            ax_.axvline(x=vline)
    fig.suptitle(df_name)
    fig.savefig(dir_save + 'plots/' + df_name + '_full.png', bbox_inches='tight')
    plt.close()

# %%
ref.to_csv(base_dir + r'Viet\ref_compare.csv', index=False)
# ref = ref[ref['ascend'] == False]

# %%
fig, ax = plt.subplots(3, 2, figsize=(16, 9))
for ax_, x, y in zip(ax.flatten(),
                     ['wdir_3s', 'ws_3s', 'temp_1m', 'rh_1m',
                      'press_1m'],
                     ['Wind Direction', 'Wind Speed', 'temp_bme', 'rh_bme',
                      'press_bme']
                     ):

    ax_.plot(ref[x], ref[y], '.')
    ax_.set_xlabel(x)
    ax_.set_ylabel(y)
    axline_min = np.min((np.min(ref[x]), np.min(ref[y])))
    axline_max = np.min((np.max(ref[x]), np.max(ref[y])))
    ax_.axline((axline_min, axline_min),
               (axline_max, axline_max),
               color='grey', linewidth=0.5, ls='--')
    try:
        line = ref[[x, y]].copy()
        line = line.dropna()
        z = np.polyfit(line[x],
                       line[y], 1)
    except Exception:
        continue
    y_hat = np.poly1d(z)(line[x])
    ax_.plot(line[x], y_hat)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(line[y], y_hat):0.3f}$"
    ax_.text(0.05, 0.95, text, transform=ax_.transAxes,
             fontsize=10, verticalalignment='top')

fig.subplots_adjust(hspace=0.3)
fig.savefig(base_dir + r'Viet\ref_compare.png', bbox_inches='tight')

#########################################
# %% No need wind data
#########################################
ref = pd.DataFrame({})
weather_full = pd.read_csv(base_dir + r'mavic_Matt\weather.csv')
weather_full = weather_full.rename({'date_time': 'datetime'}, axis='columns')
weather_full.datetime = pd.to_datetime(weather_full.datetime)

# %%
for file in file_path:
    df_name = file.split('\\')[-1][:-4]
    df = pd.read_csv(file)
    df.dropna(how='all', inplace=True)
    df = df.rename({'date_time': 'datetime'}, axis='columns')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df[['datetime', 'press_bme']] = df[['datetime', 'press_bme']].set_index(
        'datetime').rolling('10s').median().reset_index()
    df = df.reset_index()
    weather = weather_full[weather_full.datetime.dt.date == df['datetime'][0].date()]
    weather = weather.set_index('datetime').resample('1S').bfill().reset_index()
    # temp = df.merge(weather_full, on='datetime', how='left')
    temp = df.merge(weather, on='datetime', how='left')
    temp[['wdir_3s', 'wdir_10m', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
          'temp_1m', 'dewpoint', 'ws_3s', 'ws_10m', 'ws_gust',
          'voltage']] = temp[['wdir_3s', 'wdir_10m', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
                              'temp_1m', 'dewpoint', 'ws_3s', 'ws_10m', 'ws_gust',
                              'voltage']].fillna(method='backfill', limit=60)
    i_min = np.argmin(temp.press_bme)
    temp['ascend'] = True
    temp.loc[i_min:, 'ascend'] = False

    intersec = temp[np.abs(temp.press_1m - temp.press_bme) < 0.3]
    intersec = intersec.reset_index(drop=True)
    ref = ref.append(intersec)


# %%
# ref = ref[ref['ascend'] == False]

# %%
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
for ax_, x, y in zip(ax.flatten(),
                     ['temp_1m', 'rh_1m',
                      'press_1m'],
                     ['temp_bme', 'rh_bme',
                      'press_bme']
                     ):

    ax_.plot(ref[x], ref[y], '.')
    ax_.set_xlabel(x)
    ax_.set_ylabel(y)
    axline_min = np.min((np.min(ref[x]), np.min(ref[y])))
    axline_max = np.min((np.max(ref[x]), np.max(ref[y])))
    ax_.axline((axline_min, axline_min),
               (axline_max, axline_max),
               color='grey', linewidth=0.5, ls='--')
    try:
        line = ref[[x, y]].copy()
        line = line.dropna()
        z = np.polyfit(line[x],
                       line[y], 1)
    except Exception:
        continue
    y_hat = np.poly1d(z)(line[x])
    ax_.plot(line[x], y_hat)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(line[y], y_hat):0.3f}$"
    ax_.text(0.05, 0.95, text, transform=ax_.transAxes,
             fontsize=10, verticalalignment='top')

fig.subplots_adjust(hspace=0.3)
fig.savefig(base_dir + r'Viet\ref_compare_all.png', bbox_inches='tight')
