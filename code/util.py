import os
import numpy as np
import _pickle as pk
import pandas as pd
import time
import sys

animation = "|/-\\"

class DataManager:
    def __init__(self, win=5, stock_num=2330):
        self.win = win
        self.stock_num = stock_num
        self.X = []
        self.Y = []
        self.Z = []
        # Read data from data_path
        #  name       : string,  name of data
    def add_data(self, data_path):
        print ('read data from %s...'%data_path)
        days=self.win
        data = pd.read_csv(data_path)
        data = data[pd.notnull(data['Close'])]
        
        data['Date'] = data['Date'].apply(lambda x: (x[5:7]) if x[4] != '/' else x[5:7])
        data['Date'] = data['Date'].apply(lambda x: x[0] if x[1] == '/' else (x))
        data['Date'] = data['Date'].apply(lambda x: float(x))

        print ('processing stock #%d '% self.stock_num)
        for day in range(len(data[data['stock_symbol'] == self.stock_num]) - int(days)):
            sys.stdout.write("\r" + animation[day % len(animation)])
            sys.stdout.flush()
            self.X.append(data.loc[data['stock_symbol'] == self.stock_num].values[day:day+days])
            self.Y.append(data.loc[data['stock_symbol'] == self.stock_num]['Close'].values[day+days])
            self.Z.append(data.loc[data['stock_symbol'] == self.stock_num]['Open'].values[day+days])
        print ("end!")
        print (np.array(self.X).shape)
        
    def get_data(self):
        print ('get data from dataframe...')
        return np.array(self.X), np.array(self.Y), np.array(self.Z)
