import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터



#2. 모델구성
model = Sequential()                                                #[batch(행의 크기 데이터의 갯수), timesteps(몇개씩 잘라서 쓸건지), feature(몇개 씩 훈련할건지)]
model.add(LSTM(10, input_shape=(5, 1)))                        #input_shape = 행 빼고 나머지
# units *(feature +bias + units) = params

model.add(Dense(7, activation='relu'))

model.add(Dense(1))
model.summary()


# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120       ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수)  = units *(feature +bias + units) = params
                                                                    #(10 * 10) + (1 * 10) +(1 * 10)= 120

#  dense (Dense)               (None, 7)                 77


#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 205
# Trainable params: 205
# Non-trainable params: 0
# _________________________________________________________________
