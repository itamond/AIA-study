# 사이킷런 load_digits 데이터 


import pandas as pd
import numpy as np
import datetime
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

#1. 데이터 

data_sets= load_digits()

x=data_sets.data

y=data_sets['target']
# .reshape(-1, 1)



#판다스 원핫 해보기

y = to_categorical(y) #

print(x.shape, y.shape)  #(1797, 64) (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=443,
                                                    stratify=y,
                                                    shuffle=True,
                                                    )



scaler=MinMaxScaler()
# scaler=MaxAbsScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/keras28/07/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'



# print(np.unique(y_train, return_counts=True))

# #2. 모델 구성

# model = Sequential()
# model.add(Dense(100, input_dim=64))
# model.add(Dense(80,activation='relu'))
# model.add(Dense(60,activation='relu'))
# model.add(Dense(40,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(10,activation='softmax'))

input1 = Input(shape=(64,))
dense1 = Dense(100,activation='relu')(input1)
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(80,activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(60,activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(40,activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(20,activation='relu')(drop4)
drop5 = Dropout(0.2)(dense5)
output1 = Dense(10,activation='relu')(drop5)

model = Model(inputs = input1, outputs=output1)




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc']
              )

mcp = ModelCheckpoint(monitor='val_loss',
                      mode = 'auto',
                      verbose=1,
                      save_best_only=True,
                      filepath = ''.join([filepath,'k28_07_',date,'_',filename]))

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=150,
                   verbose=1)


hist = model.fit(x_train, y_train,
                 epochs = 1000,
                 validation_split=0.2,
                 batch_size=300,
                 verbose=1,
                 callbacks=[es,mcp])


#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)
y_predict = model.predict(x_test)
y_predict=np.argmax(y_predict, axis=-1)
print(y_predict.shape)

y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_predict)
print('acc :', acc)


# acc : 0.9666666666666667 맥스앱스 스케일러 적용

# acc : 0.9694444444444444 민맥스 스케일러 적용



#acc : 0.1 드롭아웃 적용... 이거 맞아?