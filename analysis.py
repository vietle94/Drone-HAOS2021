import requests
from sklearn.cluster import DBSCAN
import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

# %%
df = preprocessing.read_data(r'D:\20220301\20220301/')
df = df[df.index > pd.to_datetime('2022-03-01 10:30:00')]
df = df.resample('1s').mean()
df = df.dropna(how='all')

# %%
df['altitude_calculated'] = 44331.5 - 4946.62 * \
    (df['P(hPa)_BME-BP5']*100) ** (0.190263)
fill = df.fillna(method='bfill')
df['altitude_calculated'][np.convolve(
    df.altitude_calculated.isna(), [1, 1, 1, 1, 1], 'same') < 2] = fill['altitude_calculated'][np.convolve(
        df.altitude_calculated.isna(), [1, 1, 1, 1, 1], 'same') < 2]
df = df[df['altitude_calculated'].notna()]
df = df.reset_index()

# %%


def gaussian(x, s):
    return 1./np.sqrt(2. * np.pi * s**2) * np.exp(-x**2 / (2. * s**2))


temp = df[['altitude_calculated', 'datetime']]
gaus = np.array([gaussian(x, 20) for x in range(-50, 50, 1)])
temp_smooth = np.convolve(gaus, temp['altitude_calculated'], 'same')
coef = np.convolve(temp_smooth, [-1, 0, 1], 'same')

# fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
# ax[0].plot(temp['altitude_calculated'], '.')
# ax[1].plot(coef)
# ax[2].plot(temp_smooth)
# ax[0].plot(temp['altitude_calculated'][np.abs(coef) < 0.3], '.')

temp['is_moving'] = np.abs(coef) > 0.3

# %%
temp2 = temp[temp['is_moving'] == False]
X = np.vstack([temp2.index.values, temp2.altitude_calculated.values])
clustering = DBSCAN(eps=10, min_samples=5, n_jobs=-1).fit(X.T)
cluster_id, cluster_count = np.unique(clustering.labels_, return_counts=True)
clustering.labels_ = [-1 if x in cluster_id[cluster_count < 180] else x for x in clustering.labels_]

# fig, ax = plt.subplots()
# p = ax.scatter(temp2.datetime, temp2.altitude_calculated, c=clustering.labels_,
#                cmap='Accent')
# fig.colorbar(p, ax=ax)

# %%
df['sub_level'] = -1
df['sub_level'][temp['is_moving'] == False] = clustering.labels_

# %%
level_alt = {}
level_replace = {}
level_id = 0
for i, grp in df.groupby(['sub_level']):
    if i == -1:
        continue
    altitude_calculated = grp['altitude_calculated'].mean()
    flag = True
    for key, val in level_alt.items():
        if (altitude_calculated < val + 3) & (
                altitude_calculated > val - 3):
            level_replace[i] = key
            flag = False
            break
    if flag:
        level_id += 1
        level_replace[i] = level_id
        level_alt[level_id] = altitude_calculated

df['level'] = df['sub_level'].replace(level_replace)
level_replace

# %%
fig, ax = plt.subplots()
p = ax.scatter(df.datetime, df.altitude_calculated, c=df['level'],
               cmap='Accent')
fig.colorbar(p, ax=ax)

# %%
api_url = "https://smear-backend.rahtiapp.fi/search/timeseries/csv?" + \
    "tablevariable=KUM_META.Tower_T_32m" + \
    "&tablevariable=KUM_META.Tower_T_16m" + \
    "&tablevariable=KUM_META.rh" + \
    "&tablevariable=KUM_META.p" + \
    "&from=2022-03-01T00%3A00%3A00.000" + \
    "&to=2022-03-01T23%3A59%3A59.999" + \
    "&quality=ANY&aggregation=NONE&interval=1"

# %%
with requests.get(api_url) as response:
    name = 'smeardata' + \
        api_url.split('from=')[1].split('T')[0].replace('-', '') + '.csv'
    with open(f'./{name}', 'wb') as f:
        f.write(response.content)
        print(f'{name} was downloaded...')

# %%
data = pd.read_csv('smeardata20220301.csv')
data['Time'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
data = data.drop(['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'], axis=1)
data.columns = [x.split('.')[1] if '.' in x else x for x in data.columns]
data.columns = [x + '_smear' if 'Time' not in x else x for x in data.columns]
data = data.rename({'Time': 'datetime'}, axis='columns')

processed_df = df[df['level'] != -1].set_index('datetime').resample('1T').mean().reset_index()

# %%
full_df = processed_df.merge(data)

# %%
full_df.columns


# %%
fig, ax = plt.subplots()
ax.plot(full_df['T(C)_AHT10-BP5'],
        full_df['T(C)_BME-BP5'], '.')
# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(full_df['datetime'], full_df['altitude_calculated'])
ax[1].plot(df['datetime'], df['T(C)_BME-BP5'])
ax[1].plot(df['datetime'], df['T(C)_AHT10-BP5'])
ax[1].plot(full_df['datetime'], full_df['Tower_T_32m_smear'])
ax[1].plot(full_df['datetime'], full_df['Tower_T_16m_smear'])

# %%
data_mean = {}
data_std = {}
for i, grp in df.groupby(['level']):
    resamp = grp.set_index('datetime').resample('1T')
    data_mean[i] = resamp.mean()
    data_mean[i] = data_mean[i].dropna(how='all')
    data_std[i] = resamp.std()
    data_std[i] = data_std[i].dropna(how='all')

# %%
data_mean
data_std
