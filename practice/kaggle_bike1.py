#케글 바이크 데이터 모델링


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error



#1. 데이터

path = './_data/kaggle_bike/'
path_save= './_save/kaggle_bike/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv = pd.read_csv(path+'test.csv',index_col=0)
submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)


train_csv = train_csv.dropna()
# print(train_csv.shape)#(10886, 11)

# print(train_csv.isnull().sum())

x= train_csv.drop(['count', 'registered', 'casual'], axis=1)  #(10886, 8)

y= train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(
    x, y,
    train_size=0.99,
    shuffle=True,
    random_state=3322
)

print(x_train.shape, x_test.shape)   #(10777, 8) (109, 8)
print(y_train.shape, y_test.shape)   #(10777,) (109,)


#2. 모델 구성

model=Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(99, activation='relu'))
model.add(Dense(111, activation='relu'))
model.add(Dense(99, activation='relu'))
model.add(Dense(71, activation='relu'))
model.add(Dense(1, activation='relu'))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train,
          epochs=300,
          batch_size=120,
          verbose=1)


#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2=r2_score(y_test, y_predict)
print('r2 :', r2)

y_submit = model.predict(test_csv)

submission['count'] = y_submit

submission.to_csv(path_save + '서부미션완료.csv')


#r2 : 0.4418304429560167
# model=Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='relu'))


