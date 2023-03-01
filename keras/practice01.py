# 1. 데이터

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[1,2],[2,3]])
y = np.array([4, 5])

# 2. 모델구성

model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=50, batch_size=1)

# 4. 평가, 예측

loss = model.evaluate(x, y)
print("loss :", loss)

result = model.predict([[6,7]])
print("[[6,7]]의 예측값 :", result)
