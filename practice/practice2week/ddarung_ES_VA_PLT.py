#따릉이 데이터를 이용하여 es va plt 넣어 모델링 해~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터

path = './_data/ddarung/'
path_save = './_save/ddarung'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path +'submission.csv', index_col=0)

# print(train_set.isnull().sum())
train_set = train_set.dropna()
# print(train_set.isnull().sum())

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.95,
    random_state=332
)


#2. 모델 구성

model = Sequential()
model.add(Dense(10, input_dim= 9))
model.add(Dense(20, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련 (얼리스토핑)

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   restore_best_weights=True,
                   patience=30,
                   verbose=1)
hist = model.fit(x_train,y_train,
          epochs=200,
          batch_size=100,
          verbose=1,
          validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측

def RMSE(a,b) :
    return np.sqrt(mean_squared_error(a,b))

loss = model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)

y_submit = model.predict(test_set)
submission['count'] = y_submit
submission.to_csv(path_save+'subsubsub.csv')


#시각화
plt.rcParams['font.family']='Malgun Gothic'
plt.figure(figsize=(9,6))
plt.title('개쩌는 그래프')
plt.plot(hist.history['loss'], c='red', marker='.',label='로쓰')
plt.plot(hist.history['val_loss'],c='blue', marker='.', label='발_로스')
plt.grid()
plt.legend()
plt.show()