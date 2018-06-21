import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional, TimeDistributed, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICE'] = '1'

global history1, history2, history3

epoch = 10
batch = 256

x_train = np.load('../data/trainX_5.npy').astype(float)
y_train = np.load('../data/trainY_5.npy').astype(float)
x_test = np.load('../data/testX_5.npy').astype(float)
y_test = np.load('../data/testY_5.npy').astype(float)

x_train1 = x_train[:, :, 0].reshape(-1, 5, 1)
x_train2 = x_train[:, :, 2].reshape(-1, 5, 1)
x_train1 = np.concatenate((x_train1, x_train2), axis = 2)

mean_x = np.mean(x_train1, axis = 0)
std_x = np.std(x_train1, axis = 0)
x_1 = (x_train1 - mean_x) / std_x
x_test1 = x_test[:, :, 0].reshape(-1, 5, 1)
x_test2 = x_train[:, :, 2].reshape(-1, 5, 1)
x_t1 = (x_test1 - mean_x) / std_x

mean_y = np.mean(y_train)
std_y = np.std(y_train)
y = (y_train - mean_y) / std_y
y_t1 = (y_test - mean_y) / std_y

def model_1():
    # model1 
    model = Sequential()
    model.add(LSTM(256, input_shape=(5,2), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(5,activation='linear'))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    history = model.fit(x_1, y,batch_size = batch, epochs = 10)
    model.evaluate(x_t1, y_t1)

def model_2():
    #model2 
    model2 = Sequential()
    model2.add(LSTM(256, return_sequences=True,input_shape=(5,2)))
    model2.add(Dropout(0.4))
    model2.add(LSTM(256))
    model2.add(Dropout(0.4))
    model2.add(Dense(1, activation='linear'))
    model2.compile(optimizer='adam',loss='mse')

    history2 = model2.fit(x_1, y,batch_size = batch, epochs = 10)
    model2.evaluate(x_t1, y_t1)

def model_3():
    #model3 (0.0047961149515735133)
    model3 = Sequential()
    model3.add(LSTM(256, return_sequences=True,input_shape=(5,2)))
    model3.add(Dropout(0.4))
    model3.add(LSTM(256))
    model3.add(Dropout(0.4))
    model3.add(Dense(64))
    model3.add(Dropout(0.4))
    model3.add(Dense(1, activation='linear'))
    model3.compile(optimizer='adam',loss='mse')#, metrics=['mse'])
    history3 = model3.fit(x_1, y,batch_size = batch, epochs = 10)
    model3.evaluate(x_t1, y_t1)

def run():
    model_1()
    model_2()
    model_3()

if __name__ == '__main__':
    run()
