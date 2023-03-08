# 1. 데이터

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array(
    [[1, 3],
     [2, 5],
     [3, 7],
     [4, 2],
     [5, 6]]
)

y = np.array([6, 7, 8, 9, 10])




# 2. 모델구성

model=Sequential()
model.add(Dense(3, input_dim=2))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss=model.evaluate(x, y)
print("loss :", loss)

result=model.predict([[6,7]])
print("[6,7]의 예측값", result)



print(x.shape)
print(y.shape)