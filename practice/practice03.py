# 1. 데이터
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array(
    [[1,2],[3,4]]
    )

y = np.array([1, 3])

# 2. 모델 구성

model = Sequential()
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

result=model.predict([[9,10]])
print("[9,10]의 예측값 :", result)


print(x.shape)