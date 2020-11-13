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
    if len(y_test.shape) > 2:
        y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2])
        y_pred = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2])
    accs = get_cos(y_test, y_pred)
    print('Median exp:' + str(np.median(accs)))

    np.random.shuffle(y_pred)
    np.random.shuffle(y_test)
    accsR = get_cos(y_test, y_pred)
    print('Median random:' + str(np.median(accsR)))
    return(accs, accsR)


def lstm_prep(winsorized, n_steps):
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
        X, y = split_sequence(i, n_steps)
        if y.shape[0] != 0:
            X_ls.append(X)
            y_ls.append(y)

    dep_vars_ls = y_ls
    indep_vars_ls =X_ls
    return dep_vars_ls, indep_vars_ls

def scaled_split(winsorized, random, n_steps):
    dep_vars_ls, indep_vars_ls = lstm_prep(winsorized, n_steps)

# normalize the dataset
    y = np.vstack(dep_vars_ls)
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = scaler.fit_transform(y)
    y = np.where(np.isnan(y),0,y)


    X1 = np.vstack(indep_vars_ls)
    y = y.reshape(X1.shape[0], 1,X1.shape[2])
    X = X1.reshape(X1.shape[0]*X1.shape[1],X1.shape[2])
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X = X1.reshape(X1.shape[0], X1.shape[1],X1.shape[2])

    X = np.where(np.isnan(X),0,X)
#[samples, timesteps, features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random)
    print(y_train.shape)
    print(X_train.shape)
    return X_train, X_test, y_train, y_test

winsorized = pd.read_csv('./winsorized.csv', index_col = 0)


def create_attentionLSTM(winsorized, random, n_steps, learn, rdo, do, activ_h, KI_h, loss_fctn, size_dense, size_lstm, epochs, batch):
    #extracting properly shaped inputs, setting seeds, setting sizes going forward
    X_train, X_test, y_train, y_test = scaled_split(winsorized, random, n_steps)
    np.random.seed(random)
    tf.compat.v1.random.set_random_seed(random)
    n_steps, n_features = X_train.shape[1], X_train.shape[2]

    #initialize input layers for functional API use (different from Sequential models, allow for the multiple inputs required by attention)
    #creating query and value branches of the model in parallel
    query_input = tf.keras.layers.Input(shape = (n_steps, n_features), dtype='int32')
    value_input = tf.keras.layers.Input(shape=(n_steps, n_features), dtype='int32')
    query_embeddings = Dense(size_dense, input_shape=(n_steps, n_features), activation=activ_h, kernel_initializer=KI_h)(query_input)
    value_embeddings = Dense(size_dense, input_shape=(n_steps, n_features), activation=activ_h, kernel_initializer=KI_h)(value_input)

    #okay to reuse layer here, not above
    lstm_layer = tf.keras.layers.LSTM(size_lstm, recurrent_dropout =rdo, dropout = do, return_sequences = True)
    query_seq_encoding = lstm_layer(query_embeddings)
    value_seq_encoding = lstm_layer(value_embeddings)
    lstm_layer2 = tf.keras.layers.LSTM(size_lstm, return_sequences = True)
    query_seq_encoding2 = lstm_layer2(query_seq_encoding)
    value_seq_encoding2 = lstm_layer2(value_seq_encoding)

    query = Model(inputs=query_input, outputs=query_seq_encoding2)
    value = Model(inputs=value_input, outputs=value_seq_encoding2)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention(25)([query_seq_encoding, value_seq_encoding])

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()([query.output, query_value_attention_seq])
    layer_2 = Dense(n_features, activation='sigmoid')(input_layer)

    model = Model([query.input, value.input], layer_2)
    model.summary()
    
    opt = tf.optimizers.Adam(lr= learn, amsgrad = True)
    model.compile(loss=loss_fctn, optimizer=opt,  metrics=['acc', 'mae', 'msle', 'mse'])
    history = model.fit(x = [X_train, X_train], y = y_train, batch_size = batch,                   
                    validation_data=([X_test, X_test], y_test),
                    epochs= epochs, verbose=0, validation_split = 0.2)
    accs, accsR = test_model(model, X_train, X_test, y_train, y_test)
   # print(history.history.keys())
    return history, accs, accsR

def test_model(model, X_train, X_test, y_train, y_test):
    y_hat = model.predict([X_test, X_test], verbose=0)
    accs, accsR = get_accsr(y_test, y_hat)
    return accs,accsR


def main():
    is_test = str(input("This is a test: "))
    if is_test == "False":
        random = int(input('Select a random seed: '))
        n_steps = int(input('Select # of steps: '))
        learn = float(input('Specify learning rate: '))
        activ_h = input('Specify dense activation function: ')
        rdo = float(input('Specify recurrent dropout: '))
        do = float(input('Specify dropout: '))
        if activ_h == 'selu':
            KI_h = 'lecun_uniform'
        elif activ_h == 'relu':
            KI_h = 'he_uniform'
        elif activ_h == 'tanh':
            KI_h = 'glorot_uniform'
        loss_fctn = input('Specify loss function: ')
        size_dense = int(input('Size of Dense layers: '))
        size_lstm = int(input('Size of LSTM layer: '))
        epochs = int(input('Epochs: '))
        batch = int(input('Batch size: '))
    else:
        random, n_steps, learn, rdo, do, activ_h, KI_h, loss_fctn, size_dense, size_lstm, epochs, batch = 5, 3, 0.001, 0.02, 0.02, 'relu', 'he_uniform', 'mean_squared_error', 200,200,30, 30

    history, accs, accsR = create_attentionLSTM(winsorized, random, n_steps, learn, rdo, do, activ_h, KI_h, loss_fctn, size_dense, size_lstm, epochs, batch)
    
    vy = history.history['val_acc'][len(history.history['val_acc']) - 10:len(history.history['val_acc'])]
    ty = history.history['acc'][len(history.history['val_acc']) - 10:len(history.history['val_acc'])]
    
    print(vy, ty)
    return
    

if __name__ == "__main__":
    main()
