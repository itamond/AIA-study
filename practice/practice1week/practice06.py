#트레인, 테스트 나누기 (x의 숫자까지 지정 나눔)

import numpy as np
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터


x = np.array([range(100),range(100,200)])

x = x.T

y = np.array([range(300, 400)])

y = y.T




x_train = x[:70]
x_test = x[70:]

y_train = y[:70]
y_test = y[70:]

print(x_test)
print(x_train)
print(y_test)
print(y_train)

#2. 모델 구성

model=Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측

loss=model.evaluate(x_test,y_test)
print("loss :", loss)

result=model.predict([[505,605]])
print("[[505,605]]의 예측값 :", result)


