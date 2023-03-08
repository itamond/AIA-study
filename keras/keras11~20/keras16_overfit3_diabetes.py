from sklearn.datasets import load_diabetes
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
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
    random_state=335,
    train_size=0.9)




#2. 모델 구성


model=Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train, epochs=200, batch_size=20, validation_split=0.2)

print(hist.history)

plt.plot(hist.history['loss'])
plt.show()