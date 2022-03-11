import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import datetime
%matplotlib qt

# %%
level = 2
data_path = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula/'
df = pd.concat([pd.read_csv(x)
                for x in glob.glob(data_path + '/**/Mean_' + str(level) + '.csv', recursive=True)],
               ignore_index=True)
df_std = pd.concat([pd.read_csv(x) for x in glob.glob(
    data_path + '/**/Std_' + str(level) + '.csv', recursive=True)],
    ignore_index=True)


# %%
df.columns

# %%
if level == 2:
    t_smear = 'Tower_T_32m_smear'
else:
    t_smear = 'Tower_T_16m_smear'
fig, ax = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(16, 9))
ax[0].errorbar(df['datetime'], df['T(C)_AHT10-BP5'], marker='.',
               fmt='--', elinewidth=1,
               yerr=df_std['T(C)_AHT10-BP5'], label='T(C)_AHT10-BP5')
ax[1].errorbar(df['datetime'], df['T(C)_BME-BP5'], marker='.',
               fmt='--', elinewidth=1,
               yerr=df_std['T(C)_BME-BP5'], label='T(C)_BME-BP5')
ax[2].errorbar(df['datetime'], df['TempdegC_OPC-BP5'], marker='.',
               fmt='--', elinewidth=1,
               yerr=df_std['TempdegC_OPC-BP5'], label='TempdegC_OPC-BP5')
for ax_ in ax.flatten():
    ax_.plot(df['datetime'], df[t_smear], '.',
             label=t_smear)
    ax_.legend()
    ax_.tick_params(axis='x', labelrotation=45)
    ax_.set_ylabel('Temperature')
    ax_.grid()
    ax_.set_ylim([0, 20])
fig.subplots_adjust(bottom=0.3)
fig.savefig(r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\Summary/Temperature_ts_level' + str(level) + '.png')

# %%
fig, ax = plt.subplots(1, 3, sharex=True, figsize=(16, 9))
ax[0].plot(df[t_smear], df['T(C)_AHT10-BP5'], '.')
ax[1].plot(df[t_smear], df['T(C)_BME-BP5'], '.')
ax[2].plot(df[t_smear], df['TempdegC_OPC-BP5'], '.')
for ax_, y_ in zip(ax.flatten(), [df['T(C)_AHT10-BP5'], df['T(C)_BME-BP5'], df['TempdegC_OPC-BP5']]):
    # ax_.set_xlim([2, 5])
    # ax_.set_ylim([2, 5])
    ax_.set_aspect('equal', 'box')
    ax_.axline((0, 0), (4, 4), color='grey', linewidth=0.5,
               ls='--')
ax[0].set_xlabel(t_smear)
ax[0].set_ylabel('T(C)_AHT10-BP5')
ax[1].set_xlabel(t_smear)
ax[1].set_ylabel('T(C)_BME-BP5')
ax[2].set_xlabel(t_smear)
ax[2].set_ylabel('TempdegC_OPC-BP5')
fig.savefig(r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\Summary/Temperature_compare_level' + str(level) + '.png')


# %%
fig, ax = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(16, 9))
ax[0].errorbar(df['datetime'], df['RH(%)_AHT10-BP5'], marker='.',
               fmt='--', elinewidth=1,
               yerr=df_std['RH(%)_AHT10-BP5'], label='RH(%)_AHT10-BP5')
ax[1].errorbar(df['datetime'], df['RH(%)_BME-BP5'], marker='.',
               fmt='--', elinewidth=1,
               yerr=df_std['RH(%)_BME-BP5'], label='RH(%)_BME-BP5')
ax[2].errorbar(df['datetime'], df['RH(%)_OPC-BP5'], marker='.',
               fmt='--', elinewidth=1,
               yerr=df_std['RH(%)_OPC-BP5'], label='RH(%)_OPC-BP5')
for ax_ in ax.flatten():
    ax_.plot(df['datetime'], df['rh_smear'], '.',
             label='rh_smear')
    ax_.legend()
    ax_.tick_params(axis='x', labelrotation=45)
    ax_.set_ylabel('RH')
    ax_.grid()
fig.subplots_adjust(bottom=0.3)
fig.savefig(r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\Summary/RH_ts_level' + str(level) + '.png')

# %%
fig, ax = plt.subplots(1, 3, sharex=True, figsize=(16, 9))
ax[0].plot(df['rh_smear'], df['RH(%)_AHT10-BP5'], '.')
ax[1].plot(df['rh_smear'], df['RH(%)_BME-BP5'], '.')
ax[2].plot(df['rh_smear'], df['RH(%)_OPC-BP5'], '.')
for ax_ in ax.flatten():
    # ax_.set_xlim([2, 5])
    # ax_.set_ylim([2, 5])
    ax_.set_aspect('equal', 'box')
    ax_.axline((0, 0), (4, 4), color='grey', linewidth=0.5,
               ls='--')
ax[0].set_xlabel('rh_smear')
ax[0].set_ylabel('RH(%)_AHT10-BP5')
ax[1].set_xlabel('rh_smear')
ax[1].set_ylabel('RH(%)_BME-BP5')
ax[2].set_xlabel('rh_smear')
ax[2].set_ylabel('RH(%)_OPC-BP5')
fig.savefig(r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\Summary/RH_compare_level' + str(level) + '.png')

# %%
fig, ax = plt.subplots(sharey=True, sharex=True, figsize=(16, 9))
ax.errorbar(df['datetime'], df['P(hPa)_BME-BP5'], marker='.',
            fmt='--', elinewidth=1,
            yerr=df_std['P(hPa)_BME-BP5'], label='P(hPa)_BME-BP5')
ax.plot(df['datetime'], df['p_smear'], '.',
        label='p_smear')
ax.legend()
ax.tick_params(axis='x', labelrotation=45)
ax.set_ylabel('Pressure')
ax.grid()
fig.subplots_adjust(bottom=0.3)
fig.savefig(r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\Summary/Pressure_ts_level' + str(level) + '.png')

# %%
fig, ax = plt.subplots(sharex=True, figsize=(16, 9))
ax.plot(df['p_smear'], df['P(hPa)_BME-BP5'], '.')
ax.set_aspect('equal', 'box')
# ax.axline((0, 0), (4, 4), color='grey', linewidth=0.5,
#            ls='--')
ax.axline((np.min(df['P(hPa)_BME-BP5']), np.min(df['P(hPa)_BME-BP5'])), slope=1, color='grey', linewidth=0.5,
          ls='--')
ax.set_xlabel('p_smear')
ax.set_ylabel('P(hPa)_BME-BP5')
fig.savefig(r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Kumpula\Summary/Pressure_compare_level' + str(level) + '.png')
