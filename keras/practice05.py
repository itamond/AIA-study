# 넘파이 리스트 슬라이싱을 이용하여,  100행 1열의 데이터를 랜덤하게 테스트 해보자


import sklearn as sl
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#1. 데이터

x=np.array([range(100), range(200,300)])

x=x.T

y=np.array([range(300,400), range(500,600)])

y=y.T


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, shuffle=True, random_state=805)

print("x_train value:",x_train.shape)
print("x_test value:",x_test.shape)
print("y_train value:",y_train.shape)
print("y_test value:",y_test.shape)





#2. 모델 구성

model=Sequential()
model.add(Dense(3,input_dim=2))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))

#3. 컴파일, 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=50, batch_size=1)

#4. 평가, 예측

loss= model.evaluate(x_test,y_test)
print("loss :", loss)

result= model.predict([[146, 346]])
print("[[146, 346]]의 예측값 :", result)