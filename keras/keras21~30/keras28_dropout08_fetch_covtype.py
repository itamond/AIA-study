import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import datetime


#다중분류, 

#1. 데이터

# datasets=fetch_covtype()
a = fetch_covtype()
x = a.data
y = a.target
#7 클래스
# print(a.DESCR) 
#(581012, 54)
#(581012,)

y = to_categorical(y)
y = np.delete(y, 0, axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    stratify=y,
    random_state=388,
)

# scaler=MinMaxScaler()
scaler=MaxAbsScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

filepath = './_save/MCP/keras28/08/'
filename = '{epoch:04d}-{val_acc:.2f}.hdf5'


# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)


#2. 모델 구성

# model=Sequential()
# model.add(Dense(108, input_dim=54))
# model.add(Dense(216, activation='relu'))
# model.add(Dense(108, activation='relu'))
# model.add(Dense(54, activation='relu'))
# model.add(Dense(7, activation='softmax'))

input1 = Input(shape=(54,))
dense1 = Dense(108,activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(216,activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(108,activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(54,activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
output1 = Dense(7,activation='softmax')(drop4)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics= ['acc'],
              )

mcp = ModelCheckpoint(monitor='val_acc',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=''.join([filepath,'k28_08_',date,'_',filename]))



es = EarlyStopping(monitor='val_acc',
                   mode='max',
                   patience=100,
                   verbose=1,
                   restore_best_weights=True,
                   )
hist = model.fit(x_train, y_train,
                 epochs = 3000,
                 batch_size = 5000,
                 callbacks=[es,mcp],
                 verbose=1,
                 validation_split=0.2,
                 )



#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_argm = np.argmax(y_test, axis=1)


acc = accuracy_score(y_argm, y_pred)
print('acc :', acc)


#배치 10k , 에포 100, 페이션스 10, 발리데이션 스플릿 0.2 랜덤= 388
#result : [0.6699721217155457, 0.7153860330581665]
# acc : 0.7153860055248144



#배치 5k , 에포 100, 페이션스 10, 발리데이션 스플릿 0.2 랜덤= 388
# result : [0.6271927952766418, 0.7318313717842102]
# acc : 0.7318313640783801


#레이어 변경 * 배치 5k , 에포 100, 페이션스 10, 발리데이션 스플릿 0.2 랜덤= 388
# 변경했더니 완전 구림  인풋이 많은만큼 많은 노드가 필요한가??

#

#acc : 0.8736607488619055

# acc : 0.9119299845959227 맥스 앱스 스케일러 적용


# acc : 0.9135908711479049  드롭아웃 적용