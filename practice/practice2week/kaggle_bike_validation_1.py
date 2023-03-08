#캐글 바이크크 데이터를 이용한 모델 구성, 발리데이션 활용

# api


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


#1. 데이터

path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col= 0)
submission = pd.read_csv(path +'sampleSubmission.csv',index_col=0)

x=train_set.drop(['count','registered', 'casual'], axis=1)
y=train_set['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.9, 
    random_state= 332)


print(x_train.shape)
#2. 모델구성

model=Sequential()
model.add(Dense(10, input_dim = 8))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,
          y_train,
          epochs= 10 ,
          batch_size= 100,
          verbose = 1,
          validation_split=0.05)


#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

def RMSE(a,b) :
    return np.sqrt(mean_squared_error(a,b))




rmse = RMSE(y_test, y_predict)

print('rmse :', rmse)
y_submit = model.predict(test_set)

submission['count']=y_submit

submission.to_csv(path_save + 'submission_pr1.csv')


