# 넘파이 리스트 슬라이싱을 이용하여,  3행 10열의 데이터를 트랜스포스 한 후 랜덤하게 추출하여 테스트 해보자


import numpy as np
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터


x=np.array([[10,11,12,13,14,15,16,17,18,19],[20,19,18,17,16,15,14,13,12,11],[0,1,2,3,4,5,6,7,8,9]])  # 10행 3열 (10,3)

x=x.T  # 3행 10열 (3, 10)
y=np.array([100,101,102,103,104,105,106,107,108,109])   # 10콤마 벡터 (10, )



print(y)


# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=111)

# print("x_train:", x_train)
# print("x_test:", x_test)
# print("y_train:", y_train)
# print("y_test:", y_test)


#2. 모델 구성

model=Sequential()
model.add(Dense(4,input_dim=3))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print("loss :", loss)

result= model.predict([[20,10,10]])
print("[[20,10,10]]의 예측값 :", result)