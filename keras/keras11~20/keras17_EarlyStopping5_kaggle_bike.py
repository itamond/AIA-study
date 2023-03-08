#캐글 바이크 데이터를 사용하여
#발리데이션, 시각화, 얼리스토핑, 버보스, def 함수만들기 등을 이용한 모델 만들기 실습

import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

#1. 데이터

path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)


x= train_csv.drop(['casual','registered','count'], axis=1)
y= train_csv['count']

x_train, x_test, y_train, y_test =train_test_split(
    x, y,
    train_size=0.75,
    random_state=31
)

print(x_train.shape, x_test.shape)  #(9797, 8) (1089, 8)
print(y_train.shape, y_test.shape)


#2. 모델 구성

model=Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es=EarlyStopping(monitor='val_loss',
                 mode='min',
                 patience=50,
                 restore_best_weights=True,
                 verbose=1)

hist = model.fit(x_train,y_train,
          epochs=1000,
          batch_size=80,
          verbose=1,
          callbacks=[es],
          validation_split=0.15)


#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2=r2_score(y_test, y_predict)
print('r2 :', r2)

def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)

y_submit = model.predict(test_csv)
submission['count'] = y_submit
submission.to_csv(path_save + 'submissionES1.csv')


# #모델링
# plt.rcParams['font.family']='Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.title='쩌는 캐글 바이크'
# plt.plot(hist.history['loss'], c='red', label='로쓰', marker='.')
# plt.plot(hist.history['val_loss'], c='blue', label='발_로쓰', marker='.')
# plt.grid()
# plt.legend()
# plt.show()


# loss : 22408.587890625
# r2 : 0.3177947844454937
# rmse : 149.69498878004063


# loss : 21219.5625
# r2 : 0.31286170079867437
# rmse : 145.66935905955194