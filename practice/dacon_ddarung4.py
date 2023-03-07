#데이콘 따릉이 모델링


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터

path = './_data/ddarung/'
train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv',index_col=0)
submission = pd.read_csv(path+'submission.csv')

# print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
# print(train_csv.isnull().sum())

# print(train_csv.shape)

x= train_csv.drop('count', axis=1)

y= train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.99,
    shuffle=True,
    random_state=3837
)


###################################################

#2. 모델 구성

model=Sequential()

model.add(Dense(15, input_dim=9))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100 , batch_size=10, verbose = 1)

#4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict= model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse=RMSE(y_test, y_predict)
print('rmse :', rmse)

#submit 만들기
submit = model.predict(test_csv)
submission['count'] = submit

print(submission.shape)

submission.to_csv(path + 'submission334.csv')