import os
import numpy as np
import _pickle as pk
import pandas as pd
import time
import sys

animation = "|/-\\"

class DataManager:
    def __init__(self):
        self.data = {}
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        self.Z_train = []
        self.Z_test = []
        # Read data from data_path
        #  name       : string,  name of data
    def add_data(self, name, data_path, days=5, with_label=True):
        print ('read data from %s...'%data_path)
        X, Y, Z = [], [] , []
        data = pd.read_csv(data_path)
        data = data[pd.notnull(data['Close'])]
        print (data)
        
        data['Date'] = data['Date'].apply(lambda x: (x[5:7]) if x[4] != '/' else x[5:7])
        data['Date'] = data['Date'].apply(lambda x: x[0] if x[1] == '/' else (x))
        data['Date'] = data['Date'].apply(lambda x: float(x))

        print (data)

        for stock_num in set(data['stock_symbol'].values):
            if stock_num == 2330 or stock_num == 2317 or stock_num == 2409:
                print ('processing stock #%d '%stock_num)
                for day in range(len(data.loc[data['stock_symbol'] == stock_num]) - days):
                    sys.stdout.write("\r" + animation[day % len(animation)])
                    sys.stdout.flush()
                    X.append(data.loc[data['stock_symbol'] == stock_num].values[day:day+days])
                    Y.append(data.loc[data['stock_symbol'] == stock_num]['Close'].values[day+days])
                    Z.append(data.loc[data['stock_symbol'] == stock_num]['Open'].values[day+days])
                print ("end!")
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        
        self.data = {'train': X, 'label': Y, 'Open': Z}
        print (sum(np.isnan(X)))

    def get_data(self, name):
        print ('get data from dataframe...')
        return self.data[name]

    def split_data(self):
        print ('splitting data...')
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['train'],
        #        self.data['label'], test_size=0.1, random_state=47)
        self.X_test = self.data['train'][-49:-1]
        self.Y_test = self.data['label'][-49:-1]
        self.Z_test = self.data['Open'][-49:-1]

        self.X_train = self.data['train'][:]
        self.Y_train = self.data['label'][:]
        self.Z_train = self.data['Open'][:]

    def save_data(self, name):
        print ('saving data...')
        np.save('data/trainX_' + name + '.npy', self.X_train)
        np.save('data/trainY_' + name + '.npy', self.Y_train)
        np.save('data/testX_'  + name + '.npy', self.X_test)
        np.save('data/testY_'  + name + '.npy', self.Y_test)
        np.save('data/trainZ_' + name + '.npy', self.Z_train)
        np.save('data/testZ_' + name + '.npy', self.Z_train)

# testing
if __name__ == '__main__': 
    dm = DataManager()
    num = sys.argv[1]
    data = dm.add_data(name=num, data_path='code/data/data.csv', days=int(num))

    dm.split_data()
    dm.save_data(num)

'''
    def add_data(self, name, data_path, days=5, with_label=True):
        print ('read data from %s...'%data_path)
        X,  Y = [],  []
        data = pd.read_csv(data_path)
        
        data['Date'] = data['Date'].apply(lambda x: (x[5:7]) if x[4] != '/'else x[5:7])
        data['Date'] = data['Date'].apply(lambda x: x[0] if x[1] == '/' else (x))
        data['Date'] = data['Date'].apply(lambda x: float(x))

        for stock_num in set(data['stock_symbol'].values):
            X.append(data.loc[data['stock_symbol'] == stock_num].values[:-1])
            Y.append(data.loc[data['stock_symbol'] == stock_num]['Open'].values[1:])
        X = np.array(X)
        Y = np.array(Y)
        
        self.data[name] = np.array([X, Y])

'''
