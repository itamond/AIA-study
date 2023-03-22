import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터



#2. 모델구성
model = Sequential()                                                #[batch(행의 크기 데이터의 갯수), timesteps(몇개씩 잘라서 쓸건지), feature(몇개 씩 훈련할건지)]
model.add(GRU(10, input_shape=(5, 1)))                        #input_shape = 행 빼고 나머지


model.add(Dense(7, activation='relu'))

model.add(Dense(1))
model.summary()

#RNN params = 3 * (다음 노드 수^2 +  다음 노드 수 * Shape 의 feature + 다음 노드수 )
#LSTM에서는 출력, 입력, 삭제 게이트라는 3개의 게이트가 존재했습니다. 반면, GRU에서는 업데이트 게이트와 리셋 게이트 두 가지 게이트만 존재합니다.

# LSTM의 경우 forget gate, input gate, output gate 3개의 gate가 있었지만, 
# GRU에서는 reset gate, update gate 2개의 gate만을 사용합니다. 
# 또한 cell state, hidden state가 합쳐져 하나의 hidden state로 표현하고 있습니다. 

# 3 * (10*10 + 10 * 1 + 10)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  gru (GRU)                   (None, 10)                390

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 475
# Trainable params: 475
# Non-trainable params: 0
# _________________________________________________________________