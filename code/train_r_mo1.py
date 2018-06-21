from __future__ import print_function
import numpy as np
from util import *
from keras.models import Model, load_model, Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional, TimeDistributed, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from keras.utils import np_utils
import pickle
import argparse
import _pickle as pk
import readline
from keras import regularizers
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import pandas as pd
import os
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
from plot import plot_conf_matrix
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICE'] = '1'

parser = argparse.ArgumentParser(description='Stock prediction')
parser.add_argument('--model')
parser.add_argument('--model_type', choices=['RT_lstm', 'RT_gru', 'RF_lstm', 'RF_gru'])
parser.add_argument('--action', choices=['train','test','class'])

'''
parser.add_argument('--train_x_path', default='data/trainX_5.npy',type=str)
parser.add_argument('--train_y_path', default='data/trainY_5.npy',type=str)
parser.add_argument('--test_x_path', default= 'data/testX_5.npy' ,type=str)
parser.add_argument('--test_y_path', default= 'data/testY_5.npy' , type=str)
'''
parser.add_argument('--window', default='5', type=int)

parser.add_argument('--save-model_path', default='save/model.h5', type=str)
parser.add_argument('--save_history_path', default='save/history.npy',
        type=str)

# training argument
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.8, type=float)
parser.add_argument('--max_length', default=7,type=int)

# model parameter
parser.add_argument('--loss_function', default='mse')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.1,type=float)

# for testing
parser.add_argument('--test_y', dest='test_y', type=str, default='npy/1.npy')

# output path for your prediction
parser.add_argument('--result_path', default='result.csv')

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')

# for class
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--bin_size', default=5, type=int)

args = parser.parse_args()

mode = args.action

def RT_lstm(args):
    model = Sequential()
    model.add(LSTM(args.hidden_size, input_shape=(int(args.window),4), return_sequences=True))
    model.add(LSTM(args.hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    if mode == 'class':
        model.add(Dense(args.bin_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

    else:
        model.add(Dense(5,activation='linear'))
        model.add(Dense(1,activation='linear'))
        model.compile(loss="mse", optimizer="adam")
    
    return model
    
def RT_gru(args):
    model = Sequential()
    model.add(GRU(args.hidden_size, input_shape=(int(args.window),4), return_sequences=True))
    model.add(Dropout(args.dropout_rate))
    model.add(GRU(args.hidden_size, return_sequences=True))
    model.add(Dropout(args.dropout_rate))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    if mode == 'class':
        model.add(Dense(args.bin_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

    else:
        model.add(Dense(5,activation='linear'))
        model.add(Dense(1,activation='linear'))
        model.compile(loss="mse", optimizer="adam")
    
    return model
    
def RF_lstm(args):
    model = Sequential()
    model.add(LSTM(args.hidden_size, return_sequences=True,input_shape=(int(args.window),4)))
    model.add(Dropout(args.dropout_rate))
    model.add(LSTM(args.hidden_size))
    model.add(Dropout(args.dropout_rate))
    
    if mode == 'class':
        model.add(Dense(args.bin_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

    else:
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer="adam")

    return model
    
def RF_gru(args):
    model = Sequential()
    model.add(GRU(args.hidden_size, return_sequences=True,input_shape=(int(args.window),4)))
    model.add(Dropout(args.dropout_rate))
    model.add(GRU(args.hidden_size))
    model.add(Dropout(args.dropout_rate))
    if mode == 'class':
        model.add(Dense(args.bin_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

    else:
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer="adam")

    return model
    

def main():
    index = args.index
    bin_size = args.bin_size
    acc, pre, rec, f_score = [], [], [], []

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    save_path = os.path.join(args.save_dir,args.model)
    
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)

    #####read data#####
    dm = DataManager(args.window, args.index)
    dm.add_data('data/data.csv')
    data, label, Z = dm.get_data()
    x_train, x_test, y_train, y_test = train_test_split(data, label,
            test_size=0.1) 
    x_train = x_train.astype(float)
    y_train = y_train.astype(float)
    x_test = x_test.astype(float)
    y_test = y_test.astype(float)
    
    x_last60 = data[-61:-1, :, :]
    y_last60 = label[-61:-1]
    
    x_all = np.load('data/trainX_5.npy').astype(float)
    y_all = np.load('data/trainY_5.npy').astype(float)
    
    #####preprocess data#####
    x_train1 = x_train[:, :, 0].reshape(-1, args.window, 1)
    x_train2 = x_train[:, :, 2:5].reshape(-1, args.window, 3)
    x_train1 = np.concatenate((x_train1, x_train2), axis = 2)
    
    x_all_1 = x_all[:, :, 0].reshape(-1, args.window, 1)
    x_all_2 = x_all[:, :, 2:5].reshape(-1, args.window, 3)
    x_all_1 = np.concatenate((x_all_1, x_all_2), axis = 2)
    
    mean_x = np.mean(x_all_1, axis = 0)
    std_x = np.std(x_all_1, axis = 0)
    x_1 = (x_train1 - mean_x) / std_x
    
    x_test1 = x_test[:, :, 0].reshape(-1, args.window, 1)
    x_test2 = x_test[:, :, 2:5].reshape(-1, args.window,3)
    x_test1 = np.concatenate((x_test1, x_test2), axis = 2)
    x_t1 = (x_test1 - mean_x) / std_x
    
    x_last60_1 = x_last60[:, :, 0].reshape(-1, args.window, 1)
    x_last60_2 = x_last60[:, :, 2:5].reshape(-1, args.window, 3)
    x_last60_1 = np.concatenate((x_last60_1, x_last60_2), axis = 2)
    x_60 = (x_last60_1 - mean_x) / std_x  
    
    x_train_all = (x_all_1 - mean_x) / std_x 
    
    mean_y = np.mean(y_all)
    std_y = np.std(y_all)
    y = (y_train - mean_y) / std_y
    y_t1 = (y_test - mean_y) / std_y
    y_60 = (y_last60 - mean_y) / std_y
    y_train_all = (y_all - mean_y) / std_y

    (X_train, Y_train), (X_test, Y_test) = (x_1, y), (x_t1, y_t1)
    

    # training
    if args.action == 'train':
       
        # initial model
        print ('initial model...')
        if args.model_type == 'RT_lstm':
            model = RT_lstm(args)
        elif args.model_type == 'RT_gru':
            model = RT_gru(args)
        elif args.model_type == 'RF_lstm':
            model = RF_lstm(args)
        elif args.model_type == 'RF_gru':
            model = RF_gru(args)
        print (model.summary())
    
        (X,Y) = (x_train_all, y_train_all)
        earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1,
                mode='min')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path,
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='val_loss',
                                     mode='min' )

        history = model.fit(X, Y,
                            validation_split=0.1,
                            epochs=args.nb_epoch,
                            batch_size=args.batch_size,
                            callbacks=[checkpoint, earlystopping] )

        dict_history = pd.DataFrame(history.history)
        
        print ('saving history in: ' + args.save_history_path + '...')
        dict_history.to_csv(args.save_history_path)
    # testing
    elif args.action == 'test' :
        print ('testing ...')

        print ('loading model from \"' + load_path + '\"')
        model = load_model(load_path)

        test_y = model.predict(x_60, batch_size=args.batch_size, verbose=1)
        test_y = test_y * std_y
        test_y = test_y + mean_y
        #print (x_60[:10])
        print (y_last60[:10])
        print (test_y[:10])

        np.save(args.test_y, test_y)
           
        #draw fig
        plt.plot(test_y, color='red', label='Prediction')
        plt.plot(y_last60, color='blue', label='Ground Truth')
        plt.legend(loc='upper right')
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.ylim((70,100))
        #plt.vlines(0, 0, 70, colors = "c", linestyles = "dashed")
        plt.title("Stock Prediction of #{}".format(args.index))
        figdir = 'last60/'
        figpath = os.path.join(figdir,
                'stock-{}-{}-dim-{}-win{}.pdf'.format(args.index, args.model_type, args.hidden_size, args.window))
        plt.savefig(figpath)
        print('fig save!!! ', 'stock-{}-{}-dim-{}-win{}.pdf'.format(args.index, args.model_type, args.hidden_size, args.window))
         
    elif args.action == 'class':
        # need args: index, bin_size
        # done

        # had trainX, trainY, testX, testY
        # need to turn trainY into classes
        (train_label, test_label) = (Y_train, Y_test)

        min_val = min(train_label)
        max_val = max(train_label)
        bins = [ min_val + idx * (max_val - min_val) / (bin_size - 1) for idx in range(bin_size)]
        labels = range(bin_size - 1)
        train_label = pd.cut(train_label, bins=bins, labels=labels)
        test_label = pd.cut(test_label, bins=bins, labels=labels)


        for i in range(len(train_label)):
            if train_label[i] != train_label[i]:
                train_label[i] = 0
        
        for i in range(len(test_label)):
            if test_label[i] != test_label[i]:
                test_label[i] = 0

        print ("train_label: ", train_label)
        print (train_label.shape)
        print ("test_label: ", test_label)
        print (test_label.shape)

        Y_train = np_utils.to_categorical(train_label,
                num_classes=args.bin_size)
        Y_test = np_utils.to_categorical(test_label,
                num_classes=args.bin_size)

        (X,Y) = (X_train, Y_train)
        # model def
        # initial model
        print ('initial model...')
        if args.model_type == 'RT_lstm':
            model = RT_lstm(args)
        elif args.model_type == 'RT_gru':
            model = RT_gru(args)
        elif args.model_type == 'RF_lstm':
            model = RF_lstm(args)
        elif args.model_type == 'RF_gru':
            model = RF_gru(args)
        print (model.summary())

        
        # fit
        earlystopping = EarlyStopping(monitor='val_acc', patience = 50, verbose=1,
                mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path,
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='val_acc',
                                     mode='max' )

        history = model.fit(X, Y,
                            validation_split=0.1,
                            epochs=args.nb_epoch,
                            batch_size=args.batch_size,
                            callbacks=[checkpoint, earlystopping] )

        dict_history = pd.DataFrame(history.history)
        
        print ('saving history in: ' + args.save_history_path + '...')
        dict_history.to_csv(args.save_history_path)


        # predict
        test_y = model.predict(X_test, batch_size=args.batch_size, verbose=1)
        predict = []
        for i in range(len(test_y)):
            p = np.argmax(test_y[i])
            predict.append(p)
        predict = np.array(predict)


        # draw fig
        print (predict)
        print (test_label)
        conf_matrix = confusion_matrix(test_label, predict, labels=labels)
        figdir = 'fig/'
         
        figpath = os.path.join(figdir,
                'stock-{}-bins-{}.pdf'.format(args.index, args.bin_size))
        # labels?
        # token?
        plot_conf_matrix(conf_matrix, labels, True, args.model, figpath)
        print ('saved matrix!!!')
        accuracy = accuracy_score(test_label, predict)
        precision, recall, f, _ = precision_recall_fscore_support(test_label,
                predict, average='weighted')

        acc.append(accuracy)
        pre.append(precision)
        rec.append(recall)
        f_score.append(f)

    if mode == 'class':
         with open('log/' + args.model + '-bins-' + str(bin_size) + '.csv', 'w') as f:
            f.write('accuracy,precision,recall,f-score\n')
            for idx in range(len(acc)):
                f.write('{},{},{},{}\n'.format(acc[idx], pre[idx], rec[idx], f_score[idx]))


if __name__ == '__main__':
    main()
