import requests
from sklearn.cluster import DBSCAN
import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

# %%
flight = pd.read_excel(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\FlightRecord\20220301/wind_010322_F01.xlsx')
flight['datetime'] = pd.to_datetime('2022-03-01 10:48:49') + pd.to_timedelta(flight['Flight time'])

# %%
df = pd.read_csv(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\20220301/20220301-104658_BME-BP5.csv')
df.columns = df.columns.str.replace(' ', '')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df['datetime'], df['P(hPa)'], '.')
ax[1].plot(flight['datetime'], flight['Altitude (m)'], '.')
fig.suptitle('wind_010322_F01')
ax[0].set_ylabel('Pressure')
ax[1].set_ylabel('Altitude - from flight app')
######################################################################

# %%
flight = pd.read_excel(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\FlightRecord\20220301/wind_010322_F02.xlsx')
flight['datetime'] = pd.to_datetime('2022-03-01 11:11:06') + pd.to_timedelta(flight['Flight time'])

# %%
df = pd.read_csv(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\20220301/20220301-111025_BME-BP5.csv')
df.columns = df.columns.str.replace(' ', '')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df['datetime'], df['P(hPa)'], '.')
ax[1].plot(flight['datetime'], flight['Altitude (m)'], '.')
fig.suptitle('wind_010322_F02')
ax[0].set_ylabel('Pressure')
ax[1].set_ylabel('Altitude - from flight app')
######################################################################
# %%
flight = pd.read_excel(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\FlightRecord\20220303/DJIFlightRecord_2022-03-03_11-00-58.xlsx')
flight['datetime'] = pd.to_datetime('2022-03-03 11:00:58') + pd.to_timedelta(flight['Flight time'])

# %%
df = pd.read_csv(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\20220303/20220303-105706_BME-BP5.csv')
df.columns = df.columns.str.replace(' ', '')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df['datetime'], df['P(hPa)'], '.')
# ax[0].plot(df['datetime'], df['T(C)'], '.')
ax[1].plot(flight['datetime'], flight['Altitude'], '.')
fig.suptitle('DJIFlightRecord_2022-03-03_11-00-58')
ax[0].set_ylabel('Pressure')
ax[1].set_ylabel('Altitude - from flight app')
######################################################################

# %%
flight = pd.read_excel(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\FlightRecord\20220304/DJIFlightRecord_2022-03-04_15-43-18.xlsx')
flight['datetime'] = pd.to_datetime('2022-03-04 15:43:18') + pd.to_timedelta(flight['Flight time'])

# %%
df = pd.read_csv(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\20220304/20220304-154153_BME-BP5.csv')
df.columns = df.columns.str.replace(' ', '')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df['datetime'], df['P(hPa)'], '.')
ax[1].plot(flight['datetime'], flight['Altitude'], '.')
fig.suptitle('DJIFlightRecord_2022-03-04_15-43-18')
ax[0].set_ylim([1016, 1026])
ax[0].set_ylabel('Pressure')
ax[1].set_ylabel('Altitude - from flight app')
######################################################################

# %%
flight = pd.read_excel(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\FlightRecord\20220304/DJIFlightRecord_2022-03-04_16-03-43.xlsx')
flight['datetime'] = pd.to_datetime('2022-03-04 16:03:43') + pd.to_timedelta(flight['Flight time'])

# %%
df = pd.read_csv(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\20220304/20220304-160338_BME-BP5.csv')
df.columns = df.columns.str.replace(' ', '')
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df['datetime'], df['P(hPa)'], '.')
ax[1].plot(flight['datetime'], flight['Altitude'], '.')
fig.suptitle('DJIFlightRecord_2022-03-04_16-03-43')
ax[0].set_ylabel('Pressure')
ax[1].set_ylabel('Altitude - from flight app')

# %%
df = pd.read_csv(
    r'C:\Users\vietl\Downloads\Antarctica\Antarctica\mavic_Matt\mavic_profiles_gridded/mavic_profile_gridded_20211211_20.55.csv')
weather = pd.read_csv(r'C:\Users\vietl\Downloads\Antarctica\Antarctica\mavic_Matt/weather.csv')

# %%
df.date_time = pd.to_datetime(df.date_time)
weather.date_time = pd.to_datetime(weather.date_time)

weather = weather[weather.date_time.dt.date == df['date_time'][0].date()]

# %%
df = df.set_index('date_time').rolling('10s').median().reset_index()
# weather = weather.set_index('date_time').resample('10s').bfill().reset_index()

temp = df.merge(weather, on='date_time', how='left')
temp[weather.columns] = temp[weather.columns].fillna(method='backfill', limit=60)

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(temp.date_time, temp.press_bme, '.')
ax[0].plot(temp.date_time, temp.press_1m, '.')
ax[1].plot(temp.date_time, temp.temp_bme)
ax[1].plot(temp.date_time, temp.temp_1m, '.')

# %%
fig, ax = plt.subplots()
ax.plot(temp.date_time, temp.press_1m - temp.press_bme)

# %%
temp[np.abs(temp.press_1m - temp.press_bme) < 0.2]

# %%
fig, ax = plt.subplots(sharex=True)
ax.plot(temp.date_time, temp.temp_bme)
ax.plot(temp.date_time, temp.temp_1m, '.')
