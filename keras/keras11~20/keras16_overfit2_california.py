from sklearn.datasets import fetch_california_housing
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
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
hist = model.fit(x_train, y_train, epochs=1, batch_size= 200, validation_split=0.2, verbose=1)
print(hist.history)

plt.rcParams['font.family'] = 'Nanum Gothic'
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'],color='red')
plt.show()



# [실습]
# R2 0.55~ 0.6  이상
# train_size 0.9 이하



# loss : 0.4506089687347412
# r2 스코어 : 0.6640667257834905