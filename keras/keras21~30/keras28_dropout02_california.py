

from sklearn.datasets import fetch_california_housing
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import datetime


#1. 데이터


datasets= fetch_california_housing()
x = datasets.data
y = datasets.target



x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.9, random_state=12)

scaler=MinMaxScaler()
# scaler=MaxAbsScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성

# model = Sequential()
# model.add(Dense(5, input_dim=8))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(27, activation='relu'))
# model.add(Dense(38, activation='relu'))
# model.add(Dense(42, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(8,))
dense1 = Dense(5)(input1)
dense2 = Dense(12, activation='relu')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(27, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(38, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense4)
dense5 = Dense(42, activation='relu')(drop3)
drop4 = Dropout(0.3)(dense5)
dense6 = Dense(50, activation='relu')(drop4)
drop5 = Dropout(0.3)(dense6)
dense7 = Dense(32, activation='relu')(drop5)
drop6 = Dropout(0.3)(dense7)
dense8 = Dense(24, activation='relu')(drop6)
drop7 = Dropout(0.3)(dense8)
dense9 = Dense(12, activation='relu')(drop7)
drop8 = Dropout(0.3)(dense9)
dense10 = Dense(8, activation='relu')(drop8)
output1 = Dense(1)(dense10)

model = Model(inputs=input1, outputs=output1)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/keras28/02/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'



#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

#es -> EarlyStopping에 대한 정의

es = EarlyStopping(monitor='val_loss',     #발로스를 주시할거다
                   patience=50,             #50번 참아라
                   mode='min',               #최소값으로
                   verbose=1,                  #텍스트로 출력해라
                   restore_best_weights=True     #최고의 w값을 저장해라
                   )

mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=''.join([filepath, 'k28_02_', date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=1000, batch_size= 100,
          validation_split=0.2, callbacks=[es,mcp])

#4. 평가, 예측

loss = model.evaluate(x_test,y_test)
print("loss :", loss)


y_predict= model.predict(x_test)



r2=r2_score(y_test, y_predict)
print("r2 스코어 :", r2)


#loss : 0.6160849928855896
# r2 스코어 : 0.5407027611021253

#relu, es 첨가
# loss : 0.4392976760864258
# r2 스코어 : 0.6724994342047677

#plt 이용한 시각화

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.title='캘리포니아'
# plt.grid()
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='로쓰', marker='.')
# plt.plot(hist.history['val_loss'], c='blue', label='발_로쓰', marker='.')
# plt.legend()
# plt.show()


# loss : 0.4312790036201477
# r2 스코어 : 0.6784774127262672



#loss : 0.23141787946224213
# r2 스코어 : 0.8274757606796423     스케일러 적용


#loss : 0.7295488715171814
# r2 스코어 : 0.4561143349818526     드롭아웃 적용


