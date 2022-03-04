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
    df_merged = df_merged.drop(df_merged.filter(regex='b[1-9]*').columns, axis=1)
    df_merged = df_merged.set_index('datetime')
    df_merged = df_merged.sort_index()
    return df_merged
