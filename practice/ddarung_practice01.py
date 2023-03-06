#데이콘 따릉이 문제풀이


#임포트

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터

path = './_data/ddarung/'
train_csv= pd.read_csv(path + 'train.csv', index_col=0)
test_csv= pd.read_csv(path + 'test.csv', index_col=0)

# print(train_csv.describe)  # shape : (1459, 10)
                        # type : <class 'pandas.core.frame.DataFrame'>
                        # info() : 컬런과 널 값 확인
                        # describe : [1459 rows x 10 columns]

# print(train_csv.columns)
#Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

# print(train_csv.isnull().sum())
#hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0
# dtype: int64
# 결측 데이터 정리

train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape)  #(1328, 10)




#x와 y 데이터 분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']



x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=133
)


print(x_train.shape, x_test.shape)   #(1195, 9) (133, 9)
print(y_train.shape, y_test.shape)   #(1195,) (133,)


#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim= 9))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=20, verbose=1)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)




def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)

#submission 만들기


y_submit= model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv')

submission['count']=y_submit

submission.to_csv(path+'submission123.csv')

