import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

# %%
df = preprocessing.read_data(r'C:\Users\le\OneDrive - Ilmatieteen laitos\Drone\20220301/')
df = df[df.index > pd.to_datetime('2022-03-01 10:30:00')]

# %%
df['altitude_calculated'] = 44331.5 - 4946.62 * \
    (df['P(hPa)_BME-BP5']*100) ** (0.190263)

# %%


def gaussian(x, s):
    return 1./np.sqrt(2. * np.pi * s**2) * np.exp(-x**2 / (2. * s**2))


temp = df['altitude_calculated'].reset_index()
temp.dropna(inplace=True)
gaus = np.array([gaussian(x, 10) for x in range(-30, 30, 1)])
temp_smooth = np.convolve(gaus, temp['altitude_calculated'], 'same')
coef = np.convolve(temp_smooth, [-1, 0, 1], 'same')
# coef = pywt.wavedec(np.convolve(temp, gaus, 'same'), 'haar', level=1)
# coef = np.convolve(temp, gaus, 'same')

fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
ax[0].plot(temp['altitude_calculated'], '.')
ax[1].plot(coef)
ax[2].plot(temp_smooth)
ax[0].plot(temp['altitude_calculated'][np.abs(coef) < 0.3], '.')

# %%
temp['is_moving'] = np.abs(coef) > 0.3
temp2 = temp[temp['is_moving'] == True]

# %%
temp2 = temp2.reset_index(drop=True)

# %%
difference = temp2['datetime'].diff()
space = difference > pd.Timedelta('30 second')
np.sum(space)

# %%
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(temp2['datetime'], difference.dt.total_seconds(), '.')
ax[1].plot(df['altitude_calculated'], '.')
ax[2].plot(temp['datetime'][temp['is_moving'] == False],
           temp['altitude_calculated'][temp['is_moving'] == False], '.')
for ax_ in ax.flatten():
    ax_.grid()

# %%
