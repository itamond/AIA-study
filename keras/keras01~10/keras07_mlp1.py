import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(
   [[1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 1.3],
    [7, 1.4],
    [8, 1.5],
    [9, 1.6],
    [10, 1.4]]
)
#(10,2) -> 10행 2열, 2가지 특성을 가진 10개의 데이터 
#열=피쳐=특성=컬런



#행은 데이터의 갯수, 열은 데이터의 특성
#!!!행무시, 열우선!!!

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#(10,)   10개의 데이터이다.



print(x.shape)
print(y.shape)


#2. 모델 구성

model  = Sequential()
model.add(Dense(3, input_dim=2))   #dim 2라는것은 x의 특성, x의 열의 갯수, x의 피쳐의 갯수★
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(15))
model.add(Dense(18))
model.add(Dense(24))
model.add(Dense(28))
model.add(Dense(26))
model.add(Dense(24))
model.add(Dense(22))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(14))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=125, batch_size=3)

#4. 평가, 예측

loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([[10, 1.4]])
print('[[10, 1.4]]의 예측값 :', result)

#[[10, 1.4]]의 예측값 : [[20.502373]]
#[[10, 1.4]]의 예측값 : [[19.780586]]
#[[10, 1.4]]의 예측값 : [[20.001572]]
#[[10, 1.4]]의 예측값 : [[20.002638]]



