import numpy as np
from sklearn.cluster import DBSCAN
from functools import reduce
import pandas as pd
import os
import glob


def read_data(dir):
    # dir = r'C:\Users\le\OneDrive - Ilmatieteen laitos\Drone\20220301/'
    file_path = [x for x in glob.glob(dir + '*.csv')]
    file_names = [os.path.basename(x) for x in glob.glob(dir + '*.csv')]

    df = {}
    for name, path in zip(file_names, file_path):
        name_no_ext = name.split('.')[0]
        for name_split in name_no_ext.split('_'):
            if any(c.isalpha() for c in name_split):
                if name_split in df:
                    df[name_split] = pd.concat([df[name_split],
                                                pd.read_csv(path, index_col=False)],
                                               ignore_index=True)
                else:
                    df[name_split] = pd.read_csv(path, index_col=False)

    for key, val in df.items():
        df[key].columns = df[key].columns.str.replace(' ', '')
        if 'datetime' not in df[key].columns:
            df[key]['datetime'] = pd.to_datetime(df[key]['date'] + ' ' + df[key]['time'])
            df[key].drop(['date', 'time'], axis=1, inplace=True)
        else:
            df[key]['datetime'] = pd.to_datetime(df[key]['datetime'])
        df[key].columns = [x + '_' + key if 'datetime' not in x else x for x in df[key].columns]

    df_merged = reduce(lambda left, right: pd.merge(left,
                                                    right,
                                                    on=['datetime'], how='outer'),
                       df.values())
    # df_merged = df_merged.drop(df_merged.filter(regex='b[1-9]*').columns, axis=1)
    df_merged = df_merged.set_index('datetime')
    df_merged = df_merged.sort_index()
    return df_merged


def gaussian(x, s):
    return 1./np.sqrt(2. * np.pi * s**2) * np.exp(-x**2 / (2. * s**2))


def level_detection(pressure):
    temp = pd.DataFrame({'pressure': pressure})
    gaus = np.array([gaussian(x, 20) for x in range(-50, 50, 1)])
    temp_smooth = np.convolve(gaus, temp['pressure'], 'same')
    coef = np.convolve(temp_smooth, [-1, 0, 1], 'same')
    temp['is_moving'] = np.abs(coef) > 0.035

    temp2 = temp[temp['is_moving'] == False]
    X = np.vstack([temp2.index.values, temp2['pressure'].values])
    clustering = DBSCAN(eps=10, min_samples=5, n_jobs=-1).fit(X.T)
    cluster_id, cluster_count = np.unique(clustering.labels_, return_counts=True)
    clustering.labels_ = [-1 if x in cluster_id[cluster_count < 180]
                          else x for x in clustering.labels_]

    temp['sub_level'] = -1
    temp['sub_level'][temp['is_moving'] == False] = clustering.labels_

    level_pres = {}
    level_replace = {}
    level_id = 0
    for i, grp in temp.groupby(['sub_level']):
        if i == -1:
            continue
        altitude_calculated = grp['pressure'].mean()
        flag = True
        for key, val in level_pres.items():
            if (altitude_calculated < val + 0.5) & (
                    altitude_calculated > val - 0.5):
                level_replace[i] = key
                flag = False
                break
        if flag:
            level_id += 1
            level_replace[i] = level_id
            level_pres[level_id] = altitude_calculated

    temp['level'] = temp['sub_level'].replace(level_replace)
    return temp['level']
