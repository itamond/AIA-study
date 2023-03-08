# 따릉이 파일 모델링 과정

#import
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

#1. 데이터

path = './_data/ddarung/'
train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission = pd.read_csv(path +'submission.csv')

# print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
# print(train_csv.isnull().sum())
# print(train_csv.info())    # 1328.10
# #Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

x = train_csv.drop(['count'], axis=1)
# print(x.shape) #(1328, 9)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(
    x, y,
    train_size= 0.8,
    random_state=332,
    shuffle=True
)


print(x_train.shape, x_test.shape)    #(1062, 9) (266, 9)
print(y_train.shape, y_test.shape)    #(1062,) (266,)


#2. 모델구성 
model = Sequential()
model.add(Dense(15,input_dim=9))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

#4. 평가, 예측

loss= model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict= model.predict(x_test)

r2=r2_score(y_test, y_predict)
print('r2 :', r2)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)

print('rmse =', rmse)


#submission 만들기

y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + '서부미션퀘스트완료.csv')

#r2 : 0.5347289679524467
# rmse = 58.02525269499596

