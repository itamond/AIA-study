# x는 1개
# y는 3개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



#1. 데이터
x = np.array([range(10)])
print(x.shape) # (3, 10)
x = x.T  #(10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]]) # (3, 10)

y = y.T # (10, 3)

# 예측 : [[9]] - 예상 y 값 [[10, 1.9, 0]]


#2. 모델구성

model=Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(28))
model.add(Dense(26))
model.add(Dense(24))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(3))



#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=60, batch_size=1)

#4. 평가, 예측

loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([[9]])
print('[[9]]의 예측값 :', result)


#[[9,30,210]]의 예측값 : [[ 9.572913   -0.37402493  2.9108078 ]]
#[[9,30,210]]의 예측값 : [[ 9.764099    1.8662086  -0.12544721]]
#[[9]]의 예측값 : [[ 1.0000000e+01  1.9000008e+00 -2.7045608e-06]]
#[[9]]의 예측값 : [[10.000097    1.8985415  -0.01645362]]
#[[9]]의 예측값 : [[9.9999990e+00 1.9000005e+00 7.3015690e-07]]
#[[9]]의 예측값 : [[10.081713    1.9078078  -0.07697956]]     히든 레이어 5,8,12,15,17,20,25,28,32,35,38,32,28,26,25,24,20,16,12,10,8,3 에포 100 배치 1
#[[9]]의 예측값 : [[9.984226   1.9002553  0.02163446]]  위와같음
#[[9]]의 예측값 : [[10.051708   1.9035774 -0.0771008]]