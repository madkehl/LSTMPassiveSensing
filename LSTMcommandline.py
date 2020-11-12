#basic imports
import pandas as pd
import numpy as np
import math
#ml stuff
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM
from keras.models import Model, Sequential
from attention_decoder import AttentionDecoder
#metrics + sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats, spatial


def rearrange_rows(sequence):
    pad_imp = []
    for i in sequence:
        pad_imp.append(np.array(i))
    return(pad_imp)

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

winsorized = pd.read_csv('./winsorized.csv', index_col = 0)
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
    X, y = split_sequence(i, 4)
    if y.shape[0] != 0:
        X_ls.append(X)
        y_ls.append(y)

dep_vars_ls = y_ls
indep_vars_ls =X_ls


# normalize the dataset
y = np.vstack(dep_vars_ls)
scaler = MinMaxScaler(feature_range=(0, 1))
y = scaler.fit_transform(y)
y = np.where(np.isnan(y),0,y)

X1 = np.vstack(indep_vars_ls)
X = X1.reshape(X1.shape[0]*X1.shape[1],X1.shape[2])
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

X = X.reshape(X1.shape[0], X1.shape[1],X1.shape[2])
X = np.where(np.isnan(X),0,X)
#[samples, timesteps, features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

random = 6

np.random.seed(random)
tf.compat.v1.random.set_random_seed(random)

learn = 0.0001
opt = tf.optimizers.Adam(lr= learn, amsgrad = True)

loss_fctn = 'mean_squared_error'
activ_h = 'selu'  #selu, tanh
KI_h = 'lecun_uniform'
activ_d = 'sigmoid'
n_steps, n_features = X.shape[1], X.shape[2]
do = 0.3
rdo= 0.3

model = Sequential()
model.add(LSTM(300, activation=activ_h, kernel_initializer = KI_h, return_sequences = True, recurrent_dropout = rdo, dropout = do, input_shape=(n_steps, n_features)))
model.add(LSTM(300, activation=activ_h, kernel_initializer = KI_h,  return_sequences = True, recurrent_dropout = rdo,  dropout = do, input_shape=(n_steps, n_features)))
model.add(LSTM(300, activation=activ_h, kernel_initializer = KI_h, recurrent_dropout = rdo, dropout = do,  input_shape=(n_steps, n_features)))
model.add(Dense(X.shape[2], input_shape=(n_steps, n_features), activation=activ_d))
model.compile(loss=loss_fctn, optimizer=opt,  metrics=['acc', 'mae', 'msle', 'mse'])

history = model.fit(X_train, y_train,  batch_size=25, epochs= 20, verbose=0, validation_split = 0.2)
print(history)

y_hat = model.predict([X_test, X_test], verbose=0)
accs, accsR = get_accsr(y_test, y_hat)

