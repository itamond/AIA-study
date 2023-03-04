from sklearn.datasets import load_diabetes
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)    # (442, 10) (442,)



# [실습]

# R2 0.62 이상
# train_size 0.7이상 0.9 이하


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=72,
    train_size=0.7)




#2. 모델 구성

model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=500, batch_size=1)

#4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict=model.predict(x_test)

r2=r2_score(y_test , y_predict)

print("r2 :", r2)


#r2 : 0.578275223269473
#r2 : 0.5828874782556088   랜덤 50


#r2 : 0.6234516569138124 mae
#r2 : 0.6348515447089442 mse
#r2 : 0.6199090497243523
#r2 : 0.6374555702303933 배치 5
#r2 : 0.6337452721877734 배치 1
#