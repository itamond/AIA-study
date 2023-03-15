#데이콘 와인 데이터로 드롭 아웃 모델링 


import numpy as np
import pandas as pd
import datetime
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터

path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv = pd.read_csv(path+'test.csv',index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv',index_col=0)


# print(train_csv.columns)   #[5497 rows x 13 columns]>

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
enc.fit(train_csv['type'])
train_csv['type'] = enc.transform(train_csv['type'])
test_csv['type'] = enc.transform(test_csv['type'])
# DF = DF[DF.메뉴 != '돈까스']
# [출처] [파이썬] 7. 판다스 - 특정 조건 만족하는 행 삭제하기|작성자 러닝머신
# idx = train_csv[train_csv['quality']==['3','9']]
# idx2 = test_csv[test_csv['quality']==['3','9']]

# train_csv = train_csv[train_csv.quality != '3']
# # train_csv = train_csv[train_csv.quality != '9']
# drop33 = train_csv[train_csv['quality'].str.contains('3')].index
# train_csv =train_csv.drop(drop33, inplace=True)


print(train_csv.shape)


x= train_csv.drop(['quality'], axis=1)
y= train_csv['quality']
print(np.unique(y,return_counts=True))
# test_csv = test_csv.drop(['type'], axis=1)


y = to_categorical(y)
y = y[:,3:]

# print(y.shape)

# print(x.shape)   #(5497, 12)
# print(y.shape)
# print(test_csv)
print(np.unique(y,return_counts=True))
#array([3, 4, 5, 6, 7, 8, 9], dtype=int64)
#array([  26,  186, 1788, 2416,  924,  152,    5]
# print(y.shape)   # (5497, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=42,
    stratify=y,
    shuffle=True,
    train_size=0.8
)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)



date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/keras28/12/'
filename = '{epoch:04d}-{val_acc:.2f}.hdf5'


# model = load_model('./_save/MCP/keras28/12/k28_12_0315_1030_0722-0.61.hdf5')


#2. 모델구성


input1 = Input(shape=(12,))
dense1 = Dense(200,activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(100,activation='relu')(drop1)
drop2 = Dropout(0.5)(dense2)
dense3 = Dense(200,activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(100,activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(40,activation='relu')(drop4)
dense6 = Dense(20,activation='relu')(dense5)
output1 = Dense(7,activation='softmax')(dense6)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_acc',
                   mode='auto',
                   verbose=1,
                   restore_best_weights=True,
                   patience=150)

mcp = ModelCheckpoint(monitor='val_acc',
                      mode='auto',
                      save_best_only=True,
                      verbose=1,
                      filepath=''.join([filepath,'k28_12_',date,'_',filename])
                      )


hist = model.fit(x_train,y_train,
                 epochs= 5000,
                 batch_size = 96,
                 validation_split=0.3,
                 verbose=1,
                 callbacks=[es,mcp])

# model.save('./_save/keras26_3_save_model.h5')   

#4. 평가, 예측

result = model.evaluate(x_test,y_test)
print('result :', result)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc=accuracy_score(y_test, y_predict)
print('acc :', acc)

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis=1)
y_submit += 3

submission['quality'] = y_submit
print(y_submit)
submission.to_csv(path_save+'submission_wine_DO_5.csv')

model.fit(x_train, y_train)
print(model.feature_importances_)

# print(y_submit)



