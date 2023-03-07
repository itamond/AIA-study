#다른 api를 땡겨올때는 최상단에 땡겨오는게 깔끔하다.(가독성문제)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 5, 4])

#[실습] [6]을 예측한다

model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200)

loss=model.evaluate(x,y)
print("loss :", loss)

result=model.predict([6])
print("[6]의 예측값 :", result)

