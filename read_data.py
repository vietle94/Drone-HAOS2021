import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import fnmatch
import metpy
import xarray as xr
%matplotlib qt
# %% test


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


# %%
file_list = find('Trisonica*', 'D:/HAOS2021/HAOS2021/TDrone_PTU_wind/250321')
file_list2 = find('*BME*', 'D:/HAOS2021/HAOS2021/TDrone_PTU_wind/250321')

# %%
trisonica = pd.DataFrame()
for file in file_list:
    df = pd.read_fwf(file, sep=" ", header=None, escapechar="\\")
    data = pd.DataFrame()
    data['time'] = pd.to_datetime(df[0] + ' ' + df[1])
    data['D'] = df[5]
    data['U'] = df[7]
    data['V'] = df[9]
    data['S'] = df[3]
    trisonica = trisonica.append(data)
trisonica = trisonica.reset_index(drop=True)
trisonica['windspeed'] = np.sqrt(trisonica['U']**2 + trisonica['V']**2)

# %%
pressure = pd.DataFrame()
for file in file_list2:
    df = pd.read_csv(file, header=None)
    data = pd.DataFrame()
    data['time'] = pd.to_datetime(df[0] + ' ' + df[1])
    data['P'] = df[3]
    pressure = pressure.append(data)
pressure = pressure.reset_index(drop=True)
pressure['alt'] = 44331.5 - 4946.62 * (pressure['P']*100) ** (0.190263)

# %%
drone = pressure.merge(trisonica, on='time')
drone = drone.set_index('time')

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(drone.alt, '.')
ax[1].plot(drone.windspeed, '.')

# %%
halo = xr.open_dataset(
    r'D:\HAOS2021\HAOS2021\Halo_doppler_lidar\20210325_fmi_halo-doppler-lidar-146-VAD70-wind.nc')

halo_wind = halo.wind_speed.to_dataframe(name='wind_speed')
halo_wind = halo_wind.reset_index()
halo_wind = halo_wind.drop('range', axis=1)
halo_wind = halo_wind.dropna()
halo_wind['hours'] = halo_wind['time'].astype(int)
halo_wind['minutes'] = (halo_wind['time']*60) % 60
halo_wind['seconds'] = (halo_wind['time']*3600) % 60
halo_wind['year'] = halo.attrs['year']
halo_wind['month'] = halo.attrs['month']
halo_wind['day'] = halo.attrs['day']
halo_wind['time'] = pd.to_datetime(
    halo_wind[['year', 'month', 'day', 'hours', 'minutes', 'seconds']])
halo_wind['time'] = halo_wind['time'].astype('datetime64[s]')
halo_wind['height'] = halo_wind['height'] + halo['altitude'].data
halo_wind = halo_wind[halo_wind['height'] < 200]
halo_wind = halo_wind.set_index('time')
halo_wind['height_'] = np.round(halo_wind['height'])
# %%


def count_winthin(x, delta):
    n = len(x)
    more = np.sum(x < x[n//2] + delta)
    less = np.sum(x > x[n//2] - delta)
    return more + less


# %%
alt = drone['alt']
roll = alt.rolling('30s', center=True, win_type=None)

# %%
roll_count = roll.apply(lambda x: count_winthin(x, 3))
fig, ax = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
ax[0].plot(roll_count)
ax[1].plot(alt, '.')
ax[2].plot(alt[roll_count > 45], '.')
for ax_ in ax.flatten():
    ax_.grid()

# %%
levels = alt[roll_count > 45]
levels_count = {}
n = 0
lab = []
for i, x in enumerate(levels):
    if i == 0:
        lab.append(n)
        levels_count[n] = 1
        continue
    if abs(levels[i] - levels[i-1]) > 6:
        n = n + 1
        levels_count[n] = 1
    else:
        levels_count[n] += 1
    lab.append(n)

# %%

drone['levels'] = -1
drone['levels'][roll_count > 45] = lab
drone_denoise = drone[drone.levels > 0]

# %%
fig, axes = plt.subplots(4, 1)
for (i, g), ax in zip(halo_wind.groupby('height_'), axes.flatten()):
    ax.plot(g.wind_speed, '.', label=i)
    ax.plot(drone_denoise[(drone_denoise.alt < i + 10) & (drone_denoise.alt > i - 10)].windspeed,
            '+', label=i)
    ax.legend()
