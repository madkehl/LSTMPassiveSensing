import pandas as pd
import numpy as np
import math

def drop_homogenous(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)
    return(df)

def remove_regex(df, regex_):
    df1 = df.copy()
    return(df1[df1.columns.drop(list(df1.filter(regex=regex_)))])

def filter_perc(df, perc = 0.25):
    aware = df.copy()
    aware = aware.loc[:, aware.isna().mean() < perc]
    aware = aware.loc[:, (aware==0).mean() < perc]
    aware = aware.loc[aware.isna().mean(axis = 1) < perc, :]
    aware = aware.loc[(aware==0).mean(axis = 1) < perc, :]
    return(aware)

def winsorize(col):
    new_col = []
    col_mean = np.mean(col)
    col_std = np.std(col)
    
    zscores = [((i-col_mean)/col_std) for i in (col)]
    plus3 = col_mean + 3 *(col_std)
    minus3 = col_mean - 3 *(col_std)
    
    for i, j in enumerate(col):
        if (np.abs(zscores[i]) <= 3.0):
            new_col.append(j)
        elif zscores[i] > 3.0:
            new_col.append(plus3)
        elif zscores[i] < -3.0:
            new_col.append(minus3)
        else:
            print('error')
    return(new_col)


#aware merge will strip down to 120
aware = pd.read_csv('/users/madke/downloads/aware_personality_8-4.csv', index_col = 0)

aware['localDate'] =pd.to_datetime(aware.localDate)

aware = remove_regex(aware, 'min')
aware = remove_regex(aware, 'max')
aware = remove_regex(aware, 'sum')
aware = remove_regex(aware, 'afternoon')
aware = filter_perc(aware, 0.4)   
aware = drop_homogenous(aware)


awaregb = aware.groupby('participantID')
sequences = []
for i in list(set(aware['participantID'])):
    unit = awaregb.get_group(i)
    unit = unit.sort_values(by='localDate')
    unit = unit.reset_index(drop = True)
    unit = unit.fillna(unit.mean())
    unit = unit.fillna(0)
    sequences.append(unit.reset_index(drop = False))

sequences = [i.drop(['localDate'], axis = 1) for i in sequences]
winsorized = pd.concat(sequences, axis = 0).apply(winsorize, 0).reset_index(drop = True)
winsorized['participantID'] = aware['participantID'].reset_index(drop = True)

winsorized.to_csv('./winsorized.csv')
