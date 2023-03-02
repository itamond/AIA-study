# x는 3개
# y는 2개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
print(x.shape) # (3, 10)
x = x.T  #(10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]]) # (2, 10)

y = y.T # (10, 2)

# 실습
# 예측 : [[9,30,210]] - 예상 y 값 [[10, 1.9]]

#2. 모델 구성
model = Sequential()
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
model.add(Dense(2))

#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

#4. 평가, 예측

loss=model.evaluate(x,y)
print("loss :", loss)

result=model.predict([[9,30,210]])
print("[[9,30,210]]의 예측값 :", result)

# [[9,30,210]]의 예측값 : [[9.63821   2.5660262]]   노드 3 5 7 4 2 에포 100 배치 1
# [[9,30,210]]의 예측값 : [[10.000014   1.8999909]] 노드 3 5 8 12 15 17 20 24 20 16 12 10 8 3 2 에포 100 배치 1
# [[9,30,210]]의 예측값 : [[9.981877 2.02754 ]] 노드 동일 에포 150 배치 2
# [[9,30,210]]의 예측값 : [[10.000004   1.9000065]] 노드 동일 에포 150 배치 1
# ★★★★★[[9,30,210]]의 예측값 : [[9.999995 1.900006]] 노드 동일 에포 200 배치 1★★★★★