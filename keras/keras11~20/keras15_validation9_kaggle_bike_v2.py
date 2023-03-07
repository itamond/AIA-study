#캐글 바이크 대여 데이터 모델 만들기, 발리에이션_데이터 이용한 모의 평가 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


#1. 데이터

path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# print(train_set.shape)    #(10886, 11)
train_set = train_set.dropna()

print(train_set.isnull().sum())  
x = train_set.drop(['count','registered','casual'], axis=1)
y = train_set['count']


x_val = x[8000:]
x_test = x[6000:8000]
x_train = x[:6000]
y_val =y[8000:]
y_test = y[6000:8000]
y_train = y[:6000]


#2. 모델 구성
model= Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train,y_train,epochs=100, batch_size=100, verbose=1, validation_data=(x_val,y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test, y_predict)
print('r2 :', r2)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)

y_submit = model.predict(test_set)

submission['count']=y_submit

submission.to_csv(path_save+ 'submission_v2.csv')

# loss : 39763.7734375
# r2 : 0.04677090429631281
# rmse : 199.40854746091273
