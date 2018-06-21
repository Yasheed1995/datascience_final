from __future__ import print_function
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
from util import DataManager
from plot import plot_conf_matrix
import numpy as np
import pandas as pd
import pickle
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional, TimeDistributed, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import os
import pickle

def model_generator(token):
    if token == 'LinR':
        model = LinearRegression(normalize=True) 
    elif token == 'LogR':
        model = LogisticRegression()
    elif token == 'SVM':
        model = SVC() 
    elif token == 'D-Tree':
        model = DecisionTreeClassifier() 
    elif token == 'NN':
        model = MLPClassifier() 
    elif token == 'RF':
        model = RandomForestClassifier()
    elif token == 'KMeans':
        model = KMeans()
    elif token == 'Bayes':
        model = GaussianNB()
    elif token == 'LSTM':
        model = Sequential()
        model.add(LSTM(256, return_sequences=True,input_shape=(7,2)))
        model.add(Dropout(0.4))
        model.add(LSTM(256))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam',loss='mse')
    return model

def train(data, label, token, index, bin_size=None):
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)
    if token == 'LinR':
        train_label = np.expand_dims(train_label, axis=1)
        test_label = np.expand_dims(test_label, axis=1)
        model = model_generator(token)
        model.fit(train_data, train_label)
        modeldir = os.path.join('models', 'LinR') 
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        pickle.dump(model, open(os.path.join(modeldir, 'stock-{}.pkl'.format(index)), 'wb'))
        predict = model.predict(test_data)
        error = mean_squared_error(test_label, predict)
        return error
    else:
        min_val = min(train_label)
        max_val = max(train_label)
        bins = [ min_val + idx * (max_val - min_val) / (bin_size - 1) for idx in range(bin_size)]
        labels = range(bin_size - 1)
        print (train_label)
        train_label = pd.cut(train_label, bins=bins, labels=labels)
        test_label = pd.cut(test_label, bins=bins, labels=labels)
        #model = model_generator(token)
        for i in range(len(train_label)):
            if train_label[i] != train_label[i]:
                train_label[i] = 0
        
        for i in range(len(test_label)):
            if test_label[i] != test_label[i]:
                test_label[i] = 0
        #train_data = train_data.reshape((-1, 3615, 7))
        print (train_data.shape)
        print (train_label.shape)
        print (train_label)
        print (test_label)
        model.fit(train_data, train_label)
        modeldir = os.path.join('models', token) 
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        pickle.dump(model, open(os.path.join(modeldir, 'stock-{}-bins-{}.pkl'.format(index, bin_size)), 'wb'))
        predict = model.predict(test_data)
        conf_matrix = confusion_matrix(test_label, predict, labels=labels)
        figdir = os.path.join('fig', token)
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        figpath = os.path.join(figdir, 'stock-{}-bins-{}.pdf'.format(index, bin_size))
        plot_conf_matrix(conf_matrix, labels, True, token, figpath)
        accuracy = accuracy_score(test_label, predict)
        precision, recall, f, _ = precision_recall_fscore_support(test_label, predict, average='weighted')
        return accuracy, precision, recall, f

def argument_parser(L):
    token = L[1]
    dm = DataManager()
    dm.add_data('./data.csv')
    X = dm.get_data('data')
    Y = dm.get_data('label')
    stock_num, = X.shape
    if token == 'LinR':
        error = []
    else:
        acc, pre, rec, f_score = [], [], [], []
    for idx in range(stock_num):
        data = X[idx]
        label = Y[idx]
        if token == 'LinR':
            error.append(train(data, label, token, idx))
        else:
            bin_size = int(L[2])
            _acc, _pre, _rec, _f = train(data, label, token, idx, bin_size=bin_size)
            acc.append(_acc)
            pre.append(_pre)
            rec.append(_rec)
            f_score.append(_f)
    
    if token == 'LinR':
        with open('log/LinR.csv', 'w') as f:
            f.write('error\n')
            for idx in range(len(error)):
                f.write('{}\n'.format(error[idx]))
    else:
        with open('log/' + token + '-bins-' + str(bin_size) + '.csv', 'w') as f:
            f.write('accuracy,precision,recall,f-score\n')
            for idx in range(len(acc)):
                f.write('{},{},{},{}\n'.format(acc[idx], pre[idx], rec[idx], f_score[idx]))

if __name__ == '__main__':
    argument_parser(sys.argv)
