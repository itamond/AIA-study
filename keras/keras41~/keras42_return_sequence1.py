#return_sequence = LSTM 의 아웃풋을 2차원이 아닌 다시 3차원으로 내보내는 함수

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, GRU
from tensorflow.python.keras.callbacks import EarlyStopping
#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],[20,30,40],[30,40,50],
             [40,50,60]])
y = np.array([4,5,6,7,
              8,9,10,11,
              12,13,50,60,
              70])

x_predict = np.array([50,60,70]).reshape(1, 3, 1)  #i wan't 80
x = x.reshape(13, 3, 1)


#2. 모델
model = Sequential()
model.add(LSTM(10, input_shape=(3,1), return_sequences=True))
model.add(LSTM(11, return_sequences=True))
model.add(GRU(12))
model.add(Dense(1))


model.summary()