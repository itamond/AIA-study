#독일어여서 예나라고 읽어야한다.

#VPmax 컬런같이 치우친 데이터들은 standard scaler 한방 때려주면 가운데로 모인드... VPdef역시 동일. (안먹히면 Minmax 때린다.)
#csv 파일의 컬런은 콤마 혹은 세미콜론으로 구분


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#1. 데이터
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path+'jena_climate_2009_2016.csv', index_col=0)
# print(datasets)  #컬런은 연산 가능해야한다. 따라서 date time을 연산에 사용할 수는 없음. date time은 인덱스 형태, 때문에 index_col 해준다
#[420551 rows x 14 columns]

# print(datasets.columns)    #판다스의 명령어는 잘 알아두자
# print(datasets.info()) # 결측치 없음
print(datasets.describe())

#T가 y이다. 온도

import matplotlib.pyplot as plt



scaler = MinMaxScaler()
datasets = scaler.fit_transform(datasets)
# print(datasets['T (degC)'])  # 이 데이터의 형태는 판다스 형태이다.
# # 근데 시각화 하려면 넘파이 형태로 바꿔야함
# #방법 1. 
# print(datasets['T (degC)'].values) 
# #방법 2.
# print(datasets['T (degC)'].to_numpy())    #판다스를 넘파이로 바꾸는 방법


# plt.plot(datasets['T (degC)'].values)
# plt.show()    #규칙성있는 시계열 데이터이다.


#swish 액티베이션 = relu 친구...

#rmse, r2로 계산

#그 다음날 온도 맞추기.

# 7:2:1로 나눈다. train(validation) : test : predict

#conv1d한번 쓰고 LSTM 사용 한다.

# train, test =train_test_split(datasets,
#                             train_size=0.7,
#                             shuffle=False,
#                             )


print(datasets.shape)





x, y = split_x(datasets, timesteps, 1)
print(x)
print(y)


bbb=split_x(datasets, timesteps)

print(bbb)



# x = datasets.drop(['T (degC)'], axis=1)
# y = datasets['T (degC)']





x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=False,
                                                    )


x_test, x_pred, y_test, y_pred = train_test_split(x_test, y_test,
                                                  train_size=0.67,
                                                  shuffle=False)

print(x_train.shape, x_test.shape, x_pred.shape)    #(294385, 13) (84531, 13) (41635, 13)
#타임스텝스 10으로 짜르기





