import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import requests
%matplotlib qt

# %%
date_ = '2022-03-01'
data_path = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\FlightRecord/' + \
    date_.replace('-', '') + '/'

f1 = pd.read_excel(data_path + 'wind_010322_F01.xlsx')
f2 = pd.read_excel(data_path + 'wind_010322_F02.xlsx')

# %%
f1['datetime'] = pd.to_datetime('2022-03-01 10:48:49') + pd.to_timedelta(f1['Flight time'])
f2['datetime'] = pd.to_datetime('2022-03-01 11:11:06') + pd.to_timedelta(f2['Flight time'])

df = pd.concat([f1, f2])

# %%
api_url = "https://smear-backend.rahtiapp.fi/search/timeseries/csv?" + \
    "tablevariable=KUM_META.Tower_WS_16m" + \
    "&tablevariable=KUM_META.Tower_WS_32m" + \
    "&tablevariable=KUM_META.Tower_WDIR_32m" + \
    "&tablevariable=KUM_META.Tower_WDIR_16m" + \
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
df16 = df[(df['Altitude (m)'] > 14) & (df['Altitude (m)'] < 18)]
df32 = df[(df['Altitude (m)'] > 30) & (df['Altitude (m)'] < 34)]

# %%
resamp32 = df32.set_index('datetime').resample('1T')
mean32 = resamp32.mean()
mean32 = mean32.dropna(how='all')
std32 = resamp32.std()
std32 = std32.dropna(how='all')

# %%
std32 = std32.reset_index().merge(data)
mean32 = mean32.reset_index().merge(data)
mean32 = mean32.set_index(
    'datetime').loc[std32.set_index('datetime').index].reset_index()

# %%
fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(mean32['datetime'], mean32['Wind Speed (m/s)'], marker='.',
            fmt='--', elinewidth=1,
            yerr=std32['Wind Speed (m/s)'], label='Wind Speed (m/s)')
ax.plot(mean32['datetime'], mean32['Tower_WS_32m_smear'], label='Tower_WS_32m_smear')
ax.legend()
ax.set_ylabel('Wind speed')
fig.savefig(data_path + '/Result/wind_speed32_ts.png')

# %%
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(mean32['Tower_WS_32m_smear'], mean32['Wind Speed (m/s)'], '.')
ax.set_xlabel('Tower_WS_32m_smear')
ax.set_ylabel('Wind Speed (m/s)')
ax.set_aspect('equal')
ax.axline((np.min(mean32['Tower_WS_32m_smear']), np.min(mean32['Tower_WS_32m_smear'])), (5, 5), color='grey', linewidth=0.5,
          ls='--')
fig.savefig(data_path + '/Result/wind_speed32_compare.png')
