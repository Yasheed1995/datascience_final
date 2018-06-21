import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional, TimeDistributed, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import os
import pickle
import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np


from keras import regularizers
from keras.models import Model, load_model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend.tensorflow_backend as K
import tensorflow as tf

import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '1'

parser = argparse.ArgumentParser(description='Stock prediction')
parser.add_argument('--model')
parser.add_argument('--action', choices=['train','test','draw'])

'''
parser.add_argument('--train_x_path', default='data/trainX_5.npy',type=str)
parser.add_argument('--train_y_path', default='data/trainY_5.npy',type=str)
parser.add_argument('--test_x_path', default= 'data/testX_5.npy' ,type=str)
parser.add_argument('--test_y_path', default= 'data/testY_5.npy' , type=str)
'''
parser.add_argument('--window', default='5', type=str)

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
args = parser.parse_args()

trainX_path ='data/trainX_'  + args.window +'.npy'
trainY_path ='data/trainY_'  + args.window +'.npy'
testX_path = 'data/testX_'   + args.window +'.npy' # build model
testY_path = 'data/testY_'   + args.window +'.npy' 

def simpleRNN(args):
    inputs = Input(shape=(5,2))

    # Embedding layer
    #embedding_inputs = Embedding(args.vocab_size,
    #                             args.embedding_dim,
    #                             trainable=True)(inputs)
    # RNN
    return_sequence = True
    dropout_rate = args.dropout_rate


    for i in range(2):
        if args.cell == 'GRU':
            RNN_cell = GRU(args.hidden_size,
                       return_sequences=return_sequence,
                       dropout=dropout_rate)
        elif args.cell == 'LSTM':
            RNN_cell = LSTM(args.hidden_size,
                        return_sequences=return_sequence,
                        dropout=dropout_rate)

    RNN_output = RNN_cell(inputs)
    RNN_output = GRU(args.hidden_size,
                       return_sequences=return_sequence,
                       dropout=dropout_rate)(RNN_output)

    #
    dis_output = TimeDistributed(Dense(1))(RNN_output)
    fla_output = Flatten()(dis_output)
    

    # DNN layer
    outputs = Dense(args.hidden_size//2,
                    activation='linear',
                    kernel_regularizer=regularizers.l2(0.1))(fla_output)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1)(outputs)

    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    adam = Adam()
    print ('compile model...')

    # compile model
    model.compile( loss=args.loss_function, optimizer=adam)

    return model

def main():
    # limit gpu memory usage
    #def get_session(gpu_fraction):
    #    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    #    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #K.set_session(get_session(args.gpu_fraction))

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    save_path = os.path.join(args.save_dir,args.model)
    
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)

    #####read data#####
    x_train = np.load(trainX_path).astype(float)
    y_train = np.load(trainY_path).astype(float)
    x_test = np.load(testX_path).astype(float)
    y_test = np.load(testY_path).astype(float)

    #####preprocess data#####
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

    (X_train, Y_train), (X_test, Y_test) = (x_1, y), (x_t1, y_t1)
    
    # initial model
    print ('initial model...')
    model = simpleRNN(args)
    print (model.summary())

    # training
    if args.action == 'train':
        (X,Y) = (X_train, Y_train)
        earlystopping = EarlyStopping(monitor='loss', patience = 3, verbose=1,
                mode='min')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path,
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='loss',
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

        test_y = model.predict(X_test, batch_size=args.batch_size, verbose=1)
        test_y = test_y * std_y
        test_y = test_y + mean_y
        print (x_test[:10])
        print (y_test[:10])
        print (test_y[:10])

        np.save(args.test_y, test_y)

if __name__ == '__main__':
    main()
