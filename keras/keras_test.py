
import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100)

loss= model.evaluate(x, y)
print("loss :", loss)

result= model.predict([6])
print("4의 예측값 :", result)