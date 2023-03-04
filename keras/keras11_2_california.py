from sklearn.datasets import fetch_california_housing
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#1. 데이터
datasets= fetch_california_housing()
x = datasets.data
y = datasets.target

#(20640, 8) (20640,)

# [실습]
# R2 0.55~ 0.6  이상
# train_size 0.9 이하

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.9, random_state=12)


#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(12))
model.add(Dense(27))
model.add(Dense(38))
model.add(Dense(42))
model.add(Dense(50))
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size= 20)

#4. 평가, 예측

loss = model.evaluate(x_test,y_test)
print("loss :", loss)


y_predict= model.predict(x_test)


# R2 결정계수

r2=r2_score(y_test, y_predict)
print("r2 스코어 :", r2)

#r2 스코어 : 0.5505870601322881
#r2 스코어 : 0.500457182784201
#r2 스코어 : 0.6629057025831651  activation=relu 적용


#**r2 스코어 : 0.5419886186838374** #배치 80

#r2 스코어 : 0.5264381552539438 레이어 1층 추가

#r2 스코어 : 0.5430092727962501 #배치 40

#******r2 스코어 : 0.5594496337318692 배치 20******

#r2 스코어 : 0.5400458434619297