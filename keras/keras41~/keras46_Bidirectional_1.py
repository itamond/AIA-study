import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape)  #(7, 3) (7,)

x = x.reshape(7, 3, 1)

print(x)

#Bidirectional은 혼자서 작동이 안됨.
#지정한 함수를 양방향으로 사용하게 해주는 함수
#Bidirectional은 RNN을 함께 써줘야한다 (wrapping 한다고 함)
#Bidirectional의 첫번째 파라미터는 rnn같은 함수이고, 두번째 파라미터는 인풋 쉐이프
#2. 모델구성
model = Sequential()
# model.add(Bidirectional(SimpleRNN(10), input_shape=(3,1)))
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(3,1))) #리턴 시퀀스 해주면 똑같이 LSTM을 재사용 할 수 있다
model.add(LSTM(10, return_sequences=True))
model.add(Bidirectional(GRU(10)))
model.add(Dense(1))
model.summary()


#  Layer (type)                Output Shape              Param #
# =================================================================
#  bidirectional (Bidirectiona  (None, 20)               240
#  l)

#  dense (Dense)               (None, 1)                 21

# =================================================================


# 50, 60, 70의 결과물 : [[81.521194]]