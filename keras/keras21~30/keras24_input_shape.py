#input_dim의 문제점 = 4차원 데이터라면? dim을 어떻게 설정할것인가
#이럴때 사용하는 input_shape


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler   


#1. 데이터

datasets = load_boston()
x= datasets.data
y= datasets['target']


x_train, x_test, y_train, y_test = train_test_split (x,y,
                                                     train_size=0.8,
                                                     random_state=333,
                                                     )

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)     #scaler의 fit의 범위가 x_train이라는 뜻. (x_train을 0~1로 변환)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)             # fit에서 변환한 비율에 맞춰서 x_test를 변환해라.
print(np.min(x_test), np.max(x_test))          #-0.00557837618540494 1.1478180091225068 실제로 범위 밖으로 빠진 데이터가 있다.






#2. 모델
model=Sequential()
# model.add(Dense(1, input_dim=13))   #2 차원 데이터에서 행을 무시한 13 열
model.add(Dense(1, input_shape=(13,)))    #마찬가지로 2차원 데이터에서 행을 무시했으므로 13 콤마로 한다. 

#데이터가 3차원이라면?(시계열 데이터)
#(1000,100,1)  ->>> input_shape=(100,1)     가장 앞이 행이다.
#데이터가 4차원이라면?(이미지 데이터)
#(60000,32,32,3)  ->>> input_shape=(32,32,3)  
#데이터를 받으면 가장 먼저 shape를 찍어봐라.   앞으로는 input_shape로 쓸 것.
#모델은 input_shape와 output만 잘 맞춰주면 잘 돌아감





#3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer='adam',)
model.fit(x_train,y_train,
          epochs=10,
          batch_size=32,
          verbose=1,
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)