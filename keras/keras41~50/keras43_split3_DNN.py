import numpy as np
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping



#1. 데이터
dataset = np.array(range(1, 101))    # 1부터 100까지
timesteps = 5                       # 5개씩 잘라라
x_predict = np.array(range(96, 106))    # 100~106 의 예상값 

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape)    #(6, 5)

# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

bb = timesteps-1
x = bbb[:, : bb]   #콤마 다음은 열에 대한 이야기 이다
# x = bbb[:, :-1]  #모든행, 마지막 열 전'까지'
y = bbb[:, -1]   #모든 행, 마지막 열
x_predict = split_x(x_predict, bb)    #(7, 4)

print(x)

print(x.shape)

# x = x.reshape(96, 4, 1)
# x_predict = x_predict.reshape(7, 4, 1)

#2. 모델 구성

model= Sequential()
model.add(Dense(64, input_shape=(4,), activation='linear'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
stt = time.time()
model.compile(loss='mse', optimizer = 'adam')
es = EarlyStopping(monitor='loss',
                   mode = 'auto',
                   patience=50,
                   restore_best_weights=True,
                   verbose=1)
hist = model.fit (x, y,
                  epochs=3000,
                  callbacks=[es],
                  )

#4. 평가, 예측

loss = model.evaluate(x, y) 
print('loss :', loss)

predict = model.predict(x_predict)

print('x_predict의 예측값 :', predict)


# LSTM 적용
# [[100.00251 ]
#  [101.01356 ]
#  [102.02527 ]
#  [103.03695 ]
#  [104.04861 ]
#  [105.061966]
#  [106.07546 ]]


# DNN 적용
#  [[100.00079 ]
#  [101.000824]
#  [102.00084 ]
#  [103.000854]
#  [104.00086 ]
#  [105.000885]
#  [106.0009  ]]