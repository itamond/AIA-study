# 넘파이 리스트 슬라이싱을 이용하여,  3행 10열의 데이터를 트랜스포스 한 후 랜덤하게 추출하여 테스트 해보자


import numpy as np
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# 1. 데이터

x = np.array([range(10),range(10, 20),range(20, 30)])

x = x.T  #(10, 3)

y = np.array([range(100, 110)])

y = y.T #(10, 1)

## 데이터 나누기

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=1005)

# 2. 모델구성

model=Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 3, 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1)

# 4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print("loss :", loss)


result= model.predict([[10, 20, 30]])
print("[[10,20,30]]의 예측값 :", result)

#[[10,20,30]]의 예측값 : [[110.00002]] 