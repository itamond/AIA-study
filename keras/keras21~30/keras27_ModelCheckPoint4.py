# 저장할때 평가 결과값, 훈련시간 등을 파일에 넣어줘


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler   


#1. 데이터

datasets = load_boston()
x= datasets.data
y= datasets['target']


x_train, x_test, y_train, y_test = train_test_split (x,y,
                                                     train_size=0.8,
                                                     random_state=333,
                                                     )


scaler = StandardScaler()

x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)            
print(np.min(x_test), np.max(x_test))      





#2. 모델



input1 = Input(shape=(13,))        
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)




#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam',)

import datetime   #시간을 저장해주는 api
date = datetime.datetime.now()     #현 시간을 date에 저장
print(date)    #2023-03-14 11:11:29.334876
date = date.strftime("%m%d_%H%M")    #strftime = 스트링 퍼 타임   #% = 뒤에붙는 값 반환하라
#%m = 월 반환, %d = 날짜 반환, %H 시 반환, %M = 분 반환
#_ = 언더바는 진짜 그냥 문자다.
print(date)    #0314_1115

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#에포라는 변수의 수치 인트값 네개를 가져와라, 발로스의 소수점 4자리까지의 숫자를 받아와라




from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True,
                   )

mcp = ModelCheckpoint(monitor='val_loss',
        mode='auto',
        verbose=1,
        save_best_only=True,
        filepath= "".join([filepath, 'k27_', date, '_',filename])
)     #""는 빈공간 만듬. .join = 뭔가를 붙인다는 개념
      #''빈공간 더하기 filepath 더하기 k27_더하기 date 더하기 '_' 더하기 filename


model.fit(x_train,y_train,
        epochs=5000,
        batch_size=32,
        verbose=1,
        validation_split=0.2,
        callbacks=[es, mcp],
        )


#4. 평가, 예측
from sklearn.metrics import r2_score
print("==================== 1. 기본 출력 ======================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
