import requests
from sklearn.cluster import DBSCAN
import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

# %%
date_ = '2022-03-03'
data_path = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula/' + \
    date_.replace('-', '') + '/'
df = preprocessing.read_data(data_path)
# df = df[df.index > pd.to_datetime('2022-03-01 10:30:00')]
df = df.resample('1s').mean()
df = df.dropna(how='all')

# %%  Check outlier
pressure = 'P(hPa)_BME-BP5'
fig, ax = plt.subplots()
ax.plot(df[pressure], '.')

# %%
df = df[df[pressure] > 800]
fill = df.fillna(method='bfill')
fill_na_check = np.convolve(df[pressure].isna(), [1, 1, 1, 1, 1], 'same') < 2
df[pressure][fill_na_check] = fill[pressure][fill_na_check]
df = df[df[pressure].notna()]
df = df.reset_index()
df['level'] = preprocessing.level_detection(df['P(hPa)_BME-BP5'])

# %%
fig, ax = plt.subplots()
ax.scatter(df.datetime, df['P(hPa)_BME-BP5'], c=df['level'])

# %%
api_url = "https://smear-backend.rahtiapp.fi/search/timeseries/csv?" + \
    "tablevariable=KUM_META.Tower_T_32m" + \
    "&tablevariable=KUM_META.Tower_T_16m" + \
    "&tablevariable=KUM_META.rh" + \
    "&tablevariable=KUM_META.p" + \
    "&tablevariable=KUM_META.PM10_TEOM" + \
    "&tablevariable=KUM_META.PM25_TEOM" + \
    "&from=" + date_ + "T00%3A00%3A00.000" + \
    "&to=" + date_ + "T23%3A59%3A59.999" + \
    "&quality=ANY&aggregation=NONE&interval=1"

# %%
with requests.get(api_url) as response:
    name = 'smeardata' + \
        api_url.split('from=')[1].split('T')[0].replace('-', '') + '.csv'
    with open(data_path + f'Result/{name}', 'wb') as f:
        f.write(response.content)
        print(f'{name} was downloaded...')

# %%
data = pd.read_csv(data_path + f'Result/{name}')
data['Time'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
data = data.drop(['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'], axis=1)
data.columns = [x.split('.')[1] if '.' in x else x for x in data.columns]
data.columns = [x + '_smear' if 'Time' not in x else x for x in data.columns]
data = data.rename({'Time': 'datetime'}, axis='columns')

# %%
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(full_df['datetime'], full_df['altitude_calculated'])
# ax[1].plot(df['datetime'], df['T(C)_BME-BP5'])
# ax[1].plot(df['datetime'], df['T(C)_AHT10-BP5'])
# ax[1].plot(full_df['datetime'], full_df['Tower_T_32m_smear'])
# ax[1].plot(full_df['datetime'], full_df['Tower_T_16m_smear'])

# %%
df_mean = {}
df_std = {}
for i, grp in df.groupby(['level']):
    resamp = grp.set_index('datetime').resample('1T')
    df_mean[i] = resamp.mean()
    df_mean[i] = df_mean[i].dropna(how='all')
    df_std[i] = resamp.std()
    df_std[i] = df_std[i].dropna(how='all')

# %%
for key, item in df_mean.items():
    if key != -1:
        save_df_mean = item.reset_index().merge(data)
        save_df_std = df_std[key].reset_index().merge(data)
        save_df_mean = save_df_mean.set_index(
            'datetime').loc[save_df_std.set_index('datetime').index].reset_index()
        save_df_mean.to_csv(data_path + 'Result/Mean_' + str(key) + '.csv', index=False)
        save_df_std.to_csv(data_path + 'Result/Std_' + str(key) + '.csv', index=False)

# %%
df.columns
temp = df.filter(regex='b[0-9]*_').dropna(axis=0)
temp

# %%
temp5m = temp.resample('5T').mean()
temp5m.iloc[0].plot()
x = np.logspace(np.log(0.35), np.log(40), 24, base=np.e)
x
x = np.logspace(np.log10(0.35), np.log10(40), 24)
x
# %%
plt.plot(x, temp5m.iloc[0])
plt.xscale('log')

# %%
temp = np.loadtxt('OPC-N3_bin-boundary.txt')
temp


# %%
date_ = '2022-03-03'
data_path = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\FlightRecord/' + \
    date_.replace('-', '') + '/'

f1 = pd.read_excel(data_path + 'wind_010322_F01.xlsx')
f2 = pd.read_excel(data_path + 'wind_010322_F02.xlsx')

# %%
f1['datetime'] = pd.to_datetime('2022-03-01 10:48:49') + pd.to_timedelta(f1['Flight time'])
f2['datetime'] = pd.to_datetime('2022-03-01 11:11:06') + pd.to_timedelta(f2['Flight time'])

f = pd.concat([f1, f2])
# f = f.set_index('datetime').resample('1T').mean().dropna(how='all')

# %%
temp = df.merge(f, on='datetime', how='outer')
temp
# %%
fig, ax = plt.subplots()
ax.plot(temp['datetime'], temp['Altitude (m)'], '.')

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(temp['datetime'], temp['Altitude'], '.')
ax[1].plot(temp['datetime'], temp['level'], '.')

# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(f['datetime'], f['Altitude'], '.')
ax[1].plot(df['datetime'], df['P(hPa)_BME-BP5'], '.')

# %%
f = pd.read_excel(data_path + 'DJIFlightRecord_2022-03-03_11-00-58.xlsx')
f['datetime'] = pd.to_datetime('2022-03-03 11:00:58') + pd.to_timedelta(f['Flight time'])
