#데이콘 따릉이 데이터 모델링하기


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터

path = './_data/ddarung/'
train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission = pd.read_csv(path+'submission.csv', index_col=0)


# print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())

# print(train_csv.info()) ,type, shape, column
x= train_csv.drop(['count'],axis=1)
y= train_csv['count']


x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.99,
                                               random_state=883)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


#2. 모델구성

model=Sequential()
model.add(Dense(15,input_dim=9))
model.add(Dense(30))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train,y_train, epochs=100 , batch_size= 10, verbose=1)

#4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict=model.predict(x_test)
r2= r2_score(y_test, y_predict)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)

##submission 만들기

y_submit = model.predict(test_csv)

submission['count']=y_submit

# print(submission)

submission.to_csv(path + 'submission어쩌구.csv')