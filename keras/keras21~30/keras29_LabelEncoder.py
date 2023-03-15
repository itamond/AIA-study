# 데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
import pandas as pd 


#1. 데이터
path='./_data/dacon_wine/'      # .=현 폴더, study    /= 하위폴더

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 
                                                     
test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0)                 

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_csv['type'])   

aaa = le.transform(train_csv['type'])
print(aaa)
print(type(aaa))  #<class 'numpy.ndarray'>
# print(np.unique(aaa, return_counts=True))
train_csv['type'] = aaa
test_csv['type'] = le.transform(test_csv['type'])
#정의->핏->트랜스폼

print(le.transform(['red','white']))    # 변환 전 값을 확인하는 법


# print(train_csv.columns)
# print(train_csv.info())    
# print(type(train_csv))   


# x = train_csv.drop(['count'], axis=1)     
# print(x)

# y = train_csv['count']
# print(y)


# x_train, x_test, y_train, y_test = train_test_split(
#     x, y,
#     shuffle=True,
#     train_size=0.7,
#     random_state=221
# )





# #2. 모델구성
# model = Sequential()
# model.add(Dense(18, input_dim=9))
# model.add(Dense(27))
# model.add(Dense(36))
# model.add(Dense(45))
# model.add(Dense(36))
# model.add(Dense(27))
# model.add(Dense(18))
# model.add(Dense(9))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train,
#           epochs= 10,
#           batch_size=5,
#           verbose=1)

# #4. 평가, 예측

# loss= model.evaluate(x_test, y_test)
# print('loss :', loss)

# # y_predict= model.predict(x_test)
# # loss가 nan으로 뜨는 이유 = 결측치가 너무 많다.

# #loss : 3015.99658203125   랜덤 777  에포 100 배치 사이즈 5

# #loss : 2920.375732421875 랜덤221   에포 100 배치 5


# print(y_test)