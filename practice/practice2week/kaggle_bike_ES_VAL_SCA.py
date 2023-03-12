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
from sklearn.preprocessing import MinMaxScaler


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
    train_size=0.8,
    random_state=130
)

print(x_train.shape, x_test.shape)  #(9797, 8) (1089, 8)
print(y_train.shape, y_test.shape)

scaler= MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
test_csv = scaler.transform(test_csv)
x_test = scaler.transform(x_test)



#2. 모델 구성

model = Sequential()
model.add(Dense(6, input_dim=8))
model.add(Dense(8, activation='relu')) #↓ 값을 전달할때 값을 조절하는 함수 activation (활성화 함수) , 다음에 전달하는 내용을 *한정*시킨다.   
model.add(Dense(6, activation='relu')) # Relu -> 0 이상의 값은 양수, 0이하의 값은 0이 된다. 항상 양수로 만드는 활성화 함수
model.add(Dense(8, activation='relu'))   # 회귀모델->선형회귀. linear는 디폴트 활성화 함수
model.add(Dense(6, activation='relu'))
model.add(Dense(8, activation='relu'))   # 회귀모델->선형회귀. linear는 디폴트 활성화 함수
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es=EarlyStopping(monitor='val_loss',
                 mode='min',
                 patience=200,
                 restore_best_weights=True,
                 verbose=1)

hist = model.fit(x_train,y_train,
          epochs=5000,
          batch_size=128,
          verbose=1,
          callbacks=[es],
          validation_split=0.2)


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
submission.to_csv(path_save + 'submissionES2.csv')


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


# loss : 21414.458984375
# r2 : 0.33140530945500246
# rmse : 146.33679914745838


# loss : 16957.0078125
# r2 : 0.3563331703984126
# rmse : 130.21906733558473


#loss : 17214.6953125
# r2 : 0.3424948068351267
# rmse : 131.20477885157035