#넘파이 리스트 슬라이싱

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1,])     #=  뒤에 콤마를 찍는다고 해서 문제가 되진 않는다.

#[실습] 넘파이 리스트의 슬라이싱
x_train = x[:7]   #[1,2,3,4,5,6,7]
x_test = x[7:]    #[8.9.10]
y_train = y[:7]   #[10,9,8,7,6,5,4]
y_test = y[7:]    #[3,2,1]

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)



# 2. 모델 구성

model=Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print("loss :", loss)

result= model.predict([10])
print("[11]의 예측값 :", result)


print(x_train)
print(y_train)
print(x_test)
print(y_test)

