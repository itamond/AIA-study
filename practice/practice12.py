#데이터 시각화 해보기

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#1. 데이터

x= np.array([1,2,3,4,5,6,7,8,9,10])
y= np.array([10,11,12,13,14,15,16,17,18,19])

x=x.T
y=y.T

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7, 
    shuffle=True, 
    random_state= 1234)

#2. 모델 구성

model=Sequential()
model.add(Dense(5,  input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=1)

#4. 평가, 예측

loss=model.evaluate(x_test, y_test)
print('loss :', loss)


y_predict= model.predict(x_test)


r2=r2_score(y_test, y_predict)

print('r2 :', r2)




plt.scatter(x, y)
plt.plot(x_test ,y_predict)
plt.show()


#파이플롯의 시각화는 x와 y의 데이터 갯수를 맞춰줘야한다.