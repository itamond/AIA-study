import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPooling2D, LSTM, SimpleRNN, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import datetime


#1. 데이터

path = './_data/kaggle_jena/'
savepath = './_save/MCP/'
mcpname = '{epoch:04d}-{val_loss:.2f}.hdf5'

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)


x = datasets.drop(['T (degC)'], axis=1)
y = datasets['T (degC)']


scaler = Normalizer()

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size = 0.7,
                                                    shuffle = False,
                                                    )

x_test, x_pred, y_test, y_pred = train_test_split(x_test, y_test,
                                                  train_size = 0.67,
                                                  shuffle = False,
                                                  )


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

timesteps = 5
def split_x(dataset, timesteps) :
    aaa=[]
    for i in range(len(dataset)-timesteps-1) :
        subset = dataset[i:(i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

y_train = y_train[timesteps+1 :]
y_test = y_test[timesteps+1 :]
y_pred = y_pred[timesteps+1 :]

print(x_train.shape, x_test.shape, x_pred.shape)
#2. 모델구성

input1 = Input(shape = (10, 13))
lstm1 = LSTM(256)(input1)
drop1 = Dropout(0.5)(lstm1)
dense1 = Dense(128,activation='swish')(drop1)
drop2 = Dropout(0.5)(dense1)
dense2 = Dense(64,activation='swish')(drop2)
drop3 = Dropout(0.5)(dense2)
dense3 = Dense(32,activation='swish')(drop3)
dense4 = Dense(16)(dense3)
output1 = Dense(1,activation='relu')(dense4)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   restore_best_weights=True,
                   verbose=1,
                   mode = 'auto',
                   )


model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train,
                 epochs=300,
                 batch_size=256,
                 callbacks=[es],
                 verbose=1,
                 validation_split=0.2,
                 )

#4.평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('mae :', result[1])

pred = model.predict(x_pred)
r2 = r2_score(y_pred, pred)

def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_pred, pred)
print('rmse :', rmse)


