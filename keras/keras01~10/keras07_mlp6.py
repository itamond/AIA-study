# x는 3개
# y는 3개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
print(x.shape) # (3, 10)
x = x.T  #(10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]]) # (3, 10)

y = y.T # (10, 3)

# 예측 : [[9,30,210]] - 예상 y 값 [[10, 1.9, 0]]


#2. 모델구성

model=Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(3))



#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측

loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([[9, 30, 210]])
print('[[9,30,210]]의 예측값 :', result)


#[[9,30,210]]의 예측값 : [[ 9.572913   -0.37402493  2.9108078 ]]
#[[9,30,210]]의 예측값 : [[ 9.764099    1.8662086  -0.12544721]]


print(x)