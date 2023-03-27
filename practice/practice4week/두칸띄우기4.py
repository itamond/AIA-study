import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, LSTM, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime


#1. 데이터

path = './_data/kaggle_jena/'
filepath = './_save/MCP/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.2f}.csv'


datasets = pd.read_csv(path+'jena_climate_2009_2016.csv', index_col=0)

# print(datasets.columns)

# Index(['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
#        'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')

x= datasets.drop(['p (mbar)'], axis = 1)
y= datasets['p (mbar)']


timesteps = 10


def split_x(datasets, timesteps) :
    aaa =[]
    for i in range(len(datasets)-timesteps) :
        subset = datasets[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=False,
                                                    )


x_test,x_pred, y_test, y_pred = train_test_split(x_test, y_test,
                                                 train_size=0.67,
                                                 shuffle=False,
                                                 )


scaler = Normalizer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train=split_x(x_train,timesteps)
x_test=split_x(x_test,timesteps)
x_pred=split_x(x_pred,timesteps)

y_train = y_train[timesteps:]
y_test = y_test[timesteps:]
y_pred = y_pred[timesteps:]


print(x_train.shape)   #(294375, 10, 13)
print(y_train.shape)   #(294375,)