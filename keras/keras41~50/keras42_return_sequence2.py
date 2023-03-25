#return_sequence = LSTM 의 아웃풋을 2차원이 아닌 다시 3차원으로 내보내는 함수
import time
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, GRU, SimpleRNN
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
# model.add(LSTM(10, return_sequences=True))
model.add(GRU(10))
model.add(Dense(10))
model.add(Dense(1))


model.summary()

#RNN에서 받는 데이터는 시계열 데이터이다.
#시계열의 연속된 데이터에 대해서 만든 모델이다.
#다만 리턴 시퀀스를 통해 다음 RNN 레이어에 던져주더라도 
#그것이 명확한 시계열 데이터라는 확증이 없기 때문에 조심해야 한다


stt = time.time()

es = EarlyStopping(monitor='loss',
                #    restore_best_weights=True,
                   patience=20,
                   verbose=1,
                   mode='auto')

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x, y,
                 epochs=5000,
                 verbose=1,
                 callbacks=[es]
                 )

ent = time.time()
#4. 평가, 예측
result = model.evaluate(x,y)
print('result :', result)
y_predict = model.predict(x_predict)
print('50, 60, 70의 결과물 :', y_predict)
print('걸린 시간 : ', np.round(ent-stt, 2))


# #50, 60, 70의 결과물 : [[[41.2132  ]
#   [51.164562]
#   [52.689846]]]
# 걸린 시간 :  575.9
