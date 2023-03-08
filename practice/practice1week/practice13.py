


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt


#1. 데이터

datasets=load_diabetes()
x=datasets.data
y=datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=332,
    shuffle=True
)


#2. 모델 구성

model=Sequential()
model.add(Dense(20, input_dim=10))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train,
          epochs=100,
          batch_size=1)

#4. 평가, 예측

loss=model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict=model.predict(x_test)

r2=r2_score(y_test,y_predict)
print('r2 :', r2)



