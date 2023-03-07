import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(
    [[1,2,3,4,5,6,7,8,9,10],
     [1,1,1,1,2, 1.3, 1.4, 1.5, 1.6, 1.4],
     [9,8,7,6,5,4,3,2,1,0]]
)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

x= x.T

# 실습
# 예측 [[10,1.4,0]]


#2. 모델 구성

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
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150, batch_size=1)

#4. 평가, 예측

loss= model.evaluate(x, y)
print("loss :", loss)

result= model.predict([[10, 1.4, 0]])
print("[[10, 1.4, 0]]의 예측값 :", result)

# [[10, 1.4, 0]]의 예측값 : [[20.022924]], 노드 3 5 7 6 5 3 1 epochs 150 batch_size 1
# [[10, 1.4, 0]]의 예측값 : [[20.02607]], 노드 3 5 8 12 8 3 1 epochs 150 batch_size 1
# [[10, 1.4, 0]]의 예측값 : [[20.000002]], 노드 3 5 8 12 15 17 20 24 20 16 12 10 8 3 1 epochs 150 batch_size 1
# [[10, 1.4, 0]]의 예측값 : [[20.000006]]
# [[10, 1.4, 0]]의 예측값 : [[19.999065]]

