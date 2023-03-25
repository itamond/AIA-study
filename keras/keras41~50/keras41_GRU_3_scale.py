import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Dense, GRU, Input
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
#맹그러~~~ 시작

x = x.reshape(13, 3, 1)
#2. 모델 구성
import time
# model = Sequential()
# model.add(GRU(84, input_length = 3, input_dim = 1))
# model.add(Dense(62, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(24, activation = 'relu'))
# model.add(Dense(26, activation = 'relu'))
# model.add(Dense(1))

input1=Input(shape=(3,1))
dense1=GRU(84,activation='relu')(input1)
dense2=Dense(62,activation='relu')(dense1)
dense3=Dense(50,activation='relu')(dense2)
dense4=Dense(32,activation='relu')(dense3)
dense5=Dense(24,activation='relu')(dense4)
dense6=Dense(26,activation='relu')(dense5)
output1=Dense(1)(dense6)
model=Model(inputs=input1, outputs=output1)



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

#50, 60, 70의 결과물 : [[77.55273]]

# 50, 60, 70의 결과물 : [[82.91907]]
