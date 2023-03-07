from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=730)

# x->  앞의 두가지에 분리  y-> 뒤의 두가지에 분리


#2. 모델구성

model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(16))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mae',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print("loss :", loss)


y_predict = model.predict(x_test)


# R2= 결정 계수

r2 = r2_score(y_test,y_predict)
print('r2스코어 : ',r2)


#predict -> 훈련시키지 않은 데이터로 predict 해야한다. 


#r2스코어 :  0.7904078020452765

#r2스코어 :  0.7973596338620206

#r2스코어 :  0.9831061633519333 train_size=0.8 random 1234

#r2스코어 :  0.9757829940377987

#r2스코어 :  0.9792923029118001

#r2스코어 :  0.9465548395651746 train_size=0.7 random 5

#r2스코어 :  0.9501073933674214

#r2스코어 :  0.9765971429445547 train_size=0.7 random 7


