#basic imports
import pandas as pd
import numpy as np
import math
#ml stuff
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM
from keras.models import Model, Sequential
#metrics + sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

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

def rearrange_rows(sequence):
    pad_imp = []
    for i in sequence:
        pad_imp.append(np.array(i))
    return(pad_imp)

def drop_homogenous(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)
    return(df)

def split_sequence(sequence, n_steps = 3):
    X, y = list(), list()
    for i in range(sequence.shape[0]):
        end_ix = i + n_steps
        if end_ix > sequence.shape[0] -1:
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
        if np.array(seq_y).shape[0] != 0:
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)       
        
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

def get_cos(dft, dfp):
    cos_val = []
    for i, j in enumerate(dft):
        p = dfp[i]
        dist = spatial.distance.cosine(j, p)
        if math.isnan(dist) == False:
            cos_val.append(1 - dist)
            
    return(cos_val)

def get_accsr(y_test, y_pred):
    
    y_test = np.array(y_test)
    accs = get_cos(y_test, y_pred)
    print('Median exp:' + str(np.median(accs)))

    np.random.shuffle(y_pred)
    np.random.shuffle(y_test)
    accsR = get_cos(y_test, y_pred)
    print('Median random:' + str(np.median(accsR)))
    return(accs, accsR)

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

winsorizedgb = winsorized.groupby('participantID')
sequences_win = []
for i in list(set(winsorized['participantID'])):
    unit = winsorizedgb.get_group(i)
    sequences_win.append(unit.reset_index(drop = False))
    
indep_vars = [i.drop(['participantID'], axis = 1) for i in sequences_win]

rear_imp = rearrange_rows(indep_vars)

X_ls = []
y_ls = []
for i in rear_imp:
    X, y = split_sequence(i)
    if y.shape[0] != 0:
        X_ls.append(X)
        y_ls.append(y)
    else:
        print(y.shape)

dep_vars_ls = y_ls
indep_vars_ls =X_ls


# normalize the dataset
y = np.vstack(dep_vars_ls)
scaler = MinMaxScaler(feature_range=(0, 1))
y = scaler.fit_transform(y)
y = np.where(np.isnan(y),0,y)

# normalize the dataset
X1 = np.vstack(indep_vars_ls)

X = X1.reshape(X1.shape[0]*X1.shape[1],X1.shape[2])
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

X = X.reshape(X1.shape[0], X1.shape[1],X1.shape[2])
X = np.where(np.isnan(X),0,X)
#[samples, timesteps, features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

random = 4

np.random.seed(random)
tf.compat.v1.random.set_random_seed(random)

learn = 0.01
opt = tf.optimizers.Adam(lr= learn, amsgrad = True)

loss_fctn = 'mean_squared_error'
activ_h = 'tanh'  #selu, tanh
KI_h = 'glorot_uniform'
activ_d = 'sigmoid'
n_steps, n_features = X.shape[1], X.shape[2]
do = 0.3
rdo= 0.3

model = Sequential()
model.add(LSTM(200, activation=activ_h, kernel_initializer = KI_h, return_sequences = True, recurrent_dropout = rdo, dropout = do, input_shape=(n_steps, n_features)))
model.add(LSTM(200, activation=activ_h, kernel_initializer = KI_h,  return_sequences = True, recurrent_dropout = rdo,  dropout = do, input_shape=(n_steps, n_features)))
model.add(LSTM(200, activation=activ_h, kernel_initializer = KI_h, recurrent_dropout = rdo, dropout = do,  input_shape=(n_steps, n_features)))
model.add(Dense(X.shape[2], input_shape=(n_steps, n_features), activation=activ_d))
model.compile(loss=loss_fctn, optimizer=opt,  metrics=['acc', 'mae', 'msle', 'mse'])

history = model.fit(X_train, y_train,  batch_size=25, epochs= 500, verbose=1, validation_split = 0.2)


y_hat = model.predict(X_test, verbose=0)
accs, accsR = get_accsr(y_test, y_hat)

