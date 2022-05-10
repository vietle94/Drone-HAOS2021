from scipy import stats
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

# %%
base_dir = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Antarctica/'
dir_mavic = base_dir + r'mavic_Matt\mavic_profiles_gridded/'
file_path = [x for x in glob.glob(dir_mavic + '*.csv') if 'cal_flight' not in x]
file_names = [os.path.basename(x) for x in glob.glob(dir_mavic + '*.csv') if 'cal_flight' not in x]

dir_wind = base_dir + r'Wind_estimates/'
file_path_wind = [x for x in glob.glob(dir_wind + '*.csv')]
file_names_wind = [os.path.basename(x) for x in glob.glob(dir_wind + '*.csv')]
file_names_wind = file_names_wind[4:]  # bad first couple of files

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
    appended_df_wind.append(df_temp)

df_wind = pd.concat(appended_df_wind, ignore_index=True)

# %%
weather_full = pd.read_csv(base_dir + r'mavic_Matt\weather.csv')
weather_full = weather_full.rename({'date_time': 'datetime'}, axis='columns')
weather_full.datetime = pd.to_datetime(weather_full.datetime)
weather_full = weather_full.drop(['wdir_10m', 'ws_10m'], axis=1)

# %%
df = pd.DataFrame({})
for i, x in enumerate(file_path):
    df_ = pd.read_csv(x)
    df_['flight_ID'] = i
    df_.dropna(how='all', inplace=True)
    i_min = np.argmin(df_.press_bme)
    df_['ascend'] = True
    df_.loc[i_min:, 'ascend'] = False
    df = df.append(df_)
# df = pd.concat([pd.read_csv(x) for x in file_path], ignore_index=True)
df = df.rename({'date_time': 'datetime'}, axis='columns')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.reset_index(drop=True)

# %%
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
df_full = df_full[df_full.datetime.dt.year != 1970]
df_full = df_full.reset_index(drop=True)
df_full = df_full[~pd.isnull(df_full['datetime'])]

# %%
for data_plot, plot_name in zip([df_full, df_full[df_full['ascend'] == True],
                                 df_full[df_full['ascend'] == False]],
                                ['full', 'ascending', 'descending']):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for ax_, x, y in zip(ax.flatten(),
                         ['temp_bme', 'rh_bme'],
                         ['temp_sht', 'rh_sht']):

        ax_.plot(data_plot[x], data_plot[y], "+",
                 ms=5, mec="k", alpha=0.01)
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
    fig.savefig(base_dir + '/Viet/summary_plots/bme_sht_' + plot_name + '.png', bbox_inches='tight')
    plt.close()


# %%
ref = pd.DataFrame({})
ref_wind = pd.DataFrame({})
for i, grp in df_full.groupby(['flight_ID']):
    grp = grp.reset_index(drop=True)
    grp[['datetime', 'press_bme']] = grp[['datetime', 'press_bme']].set_index(
        'datetime').rolling('10s').median().reset_index()
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(grp.datetime, grp.pm1, '.')
    ax[0].set_ylabel('PM1')
    ax[1].plot(grp.datetime, grp.pm25, '.')
    ax[1].set_ylabel('PM25')
    ax[2].plot(grp.datetime, grp.pm10, '.')
    ax[2].set_ylabel('PM10')
    for ax_ in ax.flatten():
        ax_.grid()
    fig.savefig(base_dir + '/Viet/daily_plots/pm_ts_' +
                str(grp.datetime.min()).replace(' ', '_').replace(':', '-') + '.png', bbox_inches='tight')

    if (grp.shape[0] < 70) | (i == 33):
        continue
    fig1, ax = plt.subplots(1, 2, figsize=(12, 6))
    particle_size = grp.columns[9:25]
    grp_ = grp[grp.press_bme < (np.nanmax(grp.press_bme) - 1)]

    bin_width = 5
    lower_bin = grp_.press_bme.min() - grp_.press_bme.min() % bin_width
    bins = np.arange(lower_bin, grp_.press_bme.max()+bin_width, bin_width)
    labels = (bins[1:] + bins[:-1])/2
    grp_particle_plot = grp_.iloc[:, 9:28].copy()
    grp_particle_plot['p_binned'] = pd.cut(
        grp_['press_bme'], bins=bins, labels=labels, include_lowest=True)

    grp_particle_plot.dropna(axis=0, how='all', inplace=True)
    grp_particle_plot = grp_particle_plot.reset_index(drop=True)

    grp_particle_plot = grp_particle_plot.groupby('p_binned').mean()
    pm_plot = grp_particle_plot.iloc[:, -3:].copy()
    pm_plot = pm_plot.reset_index()

    ax[0].plot(pm_plot.pm1, pm_plot.p_binned, label='PM1')
    ax[0].plot(pm_plot.pm25, pm_plot.p_binned, label='PM2.5')
    ax[0].plot(pm_plot.pm10, pm_plot.p_binned, label='PM10')
    ax[0].set_ylabel('Pressure bme')
    ax[0].set_xlabel('Mass concentration')
    ax[0].grid()
    ax[0].invert_yaxis()
    ax[0].legend()

    p = ax[1].pcolormesh(np.arange(particle_size.size), labels,
                         grp_particle_plot.iloc[:, :-3])
    cbar = fig1.colorbar(p, ax=ax[1])
    cbar.ax.set_ylabel('Number concentration')
    ax[1].set_xlabel('Particle size')
    ax[1].set_ylabel('Pressure')
    ax[1].set_xticklabels(particle_size[::2])
    ax[1].set_xticks((np.arange(particle_size.size) + 0.5)[::2])
    ax[1].invert_yaxis()
    fig1.savefig(base_dir + '/Viet/daily_plots/daily_pm_' +
                 str(grp.datetime.min()).replace(' ', '_').replace(':', '-') + '.png', bbox_inches='tight')

    fig2, ax = plt.subplots(3, 2, figsize=(16, 9), sharex=True)
    ax[0, 0].plot(grp.datetime, grp.press_bme, '.', label='press_bme')
    ax[0, 0].plot(grp.datetime, grp.press_1m, '.', label='press_1m_tower')
    ax[0, 0].set_ylabel('Pressure [hPa]')

    ax[0, 1].plot(grp.datetime, grp.Altitude, '.', label='Drone Altitude from wind')
    ax[0, 1].set_ylabel('Altitude [m]')

    ax[1, 0].plot(grp.datetime, grp.temp_bme, '.', label='temp_bme')
    ax[1, 0].plot(grp.datetime, grp.temp_sht, '.', label='temp_sht')
    ax[1, 0].plot(grp.datetime, grp.temp_1m, '.', label='temp_1m_tower')
    ax[1, 0].set_ylabel('Temperature [$^0C$]')

    ax[1, 1].plot(grp.datetime, grp.rh_bme, '.', label='rh_bme')
    ax[1, 1].plot(grp.datetime, grp.rh_sht, '.', label='rh_sht')
    ax[1, 1].plot(grp.datetime, grp.rh_1m, '.', label='rh_1m_tower')
    ax[1, 1].set_ylabel('Relative humidity [%]')

    ax[2, 0].plot(grp.datetime, grp['Wind Speed'], '.', label='Wind speed')
    ax[2, 0].plot(grp.datetime, grp['ws_3s'], '.', label='wind_3s_tower')
    ax[2, 0].set_ylabel('Wind Speed [m/s]')

    ax[2, 1].plot(grp.datetime, grp['Wind Direction'], '.', label='Wind direction')
    ax[2, 1].plot(grp.datetime, grp['wdir_3s'], '.', label='wind_direction_3s_tower')
    ax[2, 1].set_ylabel('Wind direction [$^0$]')

    for ax_ in ax.flatten():
        ax_.legend()
        ax_.grid()

    thres_pressure = 0.5
    grp['compare_tower_others'] = np.abs(grp.press_1m - grp.press_bme) < thres_pressure
    grp['compare_tower_wind'] = np.abs((grp.press_1m - 0.7) - grp.press_bme) < thres_pressure
    ref = ref.append(grp[grp['compare_tower_others'] == True], ignore_index=True)
    ref_wind = ref_wind.append(grp[grp['compare_tower_wind'] == True],
                               ignore_index=True)

    for vline in grp[grp['compare_tower_others']].datetime:
        for ax_ in ax.flatten()[:-2]:
            ax_.axvline(x=vline, alpha=0.2)
    for vline in grp[grp['compare_tower_wind']].datetime:
        for ax_ in ax.flatten()[-2:]:
            ax_.axvline(x=vline, alpha=0.2, c='orange')
    fig2.savefig(base_dir + '/Viet/daily_plots/daily_' +
                 str(grp.datetime.min()).replace(' ', '_').replace(':', '-') + '.png', bbox_inches='tight')
    print(str(grp.datetime.min()).replace(' ', '_').replace(':', '-'))
    plt.close('all')


# %%
for data_plot_, plot_name in zip([ref, ref[ref['ascend'] == True],
                                  ref[ref['ascend'] == False]],
                                 ['full', 'ascending', 'descending']):
    for data_plot, lab_avg in zip([data_plot_, data_plot_.groupby(['flight_ID', 'ascend']).mean()],
                                  ['', '_avg']):
        fig, ax = plt.subplots(2, 2, figsize=(16, 9))
        for ax_, x, y in zip(ax.flatten(),
                             ['temp_1m', 'rh_1m',
                              'temp_1m', 'rh_1m'],
                             ['temp_bme', 'rh_bme',
                              'temp_sht', 'rh_sht']
                             ):

            ax_.plot(data_plot[x], data_plot[y], "+",
                     ms=5, mec="k", alpha=0.5)
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

        fig.subplots_adjust(hspace=0.3)
        fig.savefig(base_dir + '/Viet/summary_plots/tem_rh_compare_' +
                    plot_name + lab_avg + '.png', bbox_inches='tight')
        # plt.close()

# %%
for data_plot_, plot_name in zip([ref_wind, ref_wind[ref_wind['ascend'] == True],
                                  ref_wind[ref_wind['ascend'] == False]],
                                 ['full', 'ascending', 'descending']):
    for data_plot, lab_avg in zip([data_plot_, data_plot_.groupby(['flight_ID', 'ascend']).mean()],
                                  ['', '_avg']):
        fig, ax = plt.subplots(2, figsize=(16, 9))
        for ax_, x, y in zip(ax.flatten(),
                             ['wdir_3s', 'ws_3s'],
                             ['Wind Direction', 'Wind Speed']
                             ):

            ax_.plot(data_plot[x], data_plot[y], "+",
                     ms=5, mec="k", alpha=0.5)
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

        fig.subplots_adjust(hspace=0.3)
        fig.savefig(base_dir + '/Viet/summary_plots/wind_compare_' +
                    plot_name + lab_avg + '.png', bbox_inches='tight')
        # plt.close()