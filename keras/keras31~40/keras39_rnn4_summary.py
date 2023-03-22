import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# y = ?

x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],
              [4,5,6,7,8],[5,6,7,8,9]])
#y값을 지정해줘야하기 때문에 10은 뺀다

y = np.array([6, 7, 8, 9, 10])

print(x.shape, y.shape)  #(5, 5) (5,)


#RNN은 통상 3차원 데이터로 훈련.
#[1,2,3] 훈련을 한다면 1한번 2한번 3한번 훈련한다

#x의 shape = (행, 열, 몇개씩 훈련하는지) 
#이번 경우에는 1개씩 훈련하기때문에 끝을 1로 리쉐이프 해줘야함.

x = x.reshape(5, 5, 1)
print(x)

#2. 모델구성
model = Sequential()                                                #[batch(행의 크기 데이터의 갯수), timesteps(몇개씩 잘라서 쓸건지), feature(몇개 씩 훈련할건지)]
model.add(SimpleRNN(10, input_shape=(5, 1)))                        #input_shape = 행 빼고 나머지
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
