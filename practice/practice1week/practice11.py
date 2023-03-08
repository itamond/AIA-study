# x= [range(10)], y = [range(10,20)] 인 데이터로 시퀀셜 모델을 만들고 pyplot을 통해 시각화해보자.(실패. 데이터의 종류가 시각화 불가능한 데이터)


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#1. 데이터 


x = np.array([range(10)])
y = np.array([range(10, 20)])

x=x.T
y=y.T



# x_train=x[:5]
# x_test=x[5:]
# y_train=y[:5]
# y_test=y[5:]


x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7,
    shuffle=True,
    random_state=1234)


#2. 모델구성

model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측

loss=model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict=model.predict(x_test)

r2=r2_score(y_test, y_predict)