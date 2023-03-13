#캐글 바이크 데이터를 사용하여
#발리데이션, 시각화, 얼리스토핑, 버보스, def 함수만들기 등을 이용한 모델 만들기 실습

import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
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

scaler=MinMaxScaler()
# scaler=MaxAbsScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)



print(x_train.shape, x_test.shape)  #(9797, 8) (1089, 8)
print(y_train.shape, y_test.shape)


#2. 모델 구성

# model = Sequential()
# model.add(Dense(6, input_dim=8))
# model.add(Dense(8, activation='relu')) #↓ 값을 전달할때 값을 조절하는 함수 activation (활성화 함수) , 다음에 전달하는 내용을 *한정*시킨다.   
#                                         # Relu -> 0 이상의 값은 양수, 0이하의 값은 0이 된다. 항상 양수로 만드는 활성화 함수
# model.add(Dense(6))   # 회귀모델->선형회귀. linear는 디폴트 활성화 함수
# model.add(Dense(12, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(1))


input1 = Input(shape=(8,))
dense1 = Dense(6)(input1)
dense2 = Dense(8,activation='relu')(dense1)
dense3 = Dense(6)(dense2)
dense4 = Dense(12,activation='relu')(dense3)
dense5 = Dense(8,activation='relu')(dense4)
dense6 = Dense(6,activation='relu')(dense5)
dense7 = Dense(8,activation='relu')(dense6)
dense8 = Dense(6,activation='relu')(dense7)
output1 = Dense(1)(dense8)


model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es=EarlyStopping(monitor='val_loss',
                 mode='min',
                 patience=100,
                 restore_best_weights=True,
                 verbose=1)

hist = model.fit(x_train,y_train,
          epochs=1000,
          batch_size=100,
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
submission.to_csv(path_save + 'submissionSC1.csv')


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



# loss : 25118.5234375
# r2 : 0.2976472817161939
# rmse : 158.48824711449768 스탠 스케일러


# loss : 26447.603515625
# r2 : 0.26048414006011333
# rmse : 162.62719620858482 로버스트 스케일러


# loss : 26730.6171875
# r2 : 0.2525706353268429
# rmse : 163.49501160178679  맥스앱스 스케일러



# loss : 22945.6015625
# r2 : 0.3253500969470541
# rmse : 151.47806049751628 민맥스 스케일러


# loss : 23524.501953125
# r2 : 0.30832910727010365
# rmse : 153.3770055629679


# loss : 22730.095703125
# r2 : 0.3316863315918993
# rmse : 150.76505011989627