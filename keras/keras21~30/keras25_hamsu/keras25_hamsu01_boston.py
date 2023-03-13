#input_dim의 문제점 = 4차원 데이터라면? dim을 어떻게 설정할것인가
#이럴때 사용하는 input_shape


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
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
# model=Sequential()
# model.add(Dense(30, input_shape=(13,))) 
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))



input1 = Input(shape=(13,))         #스칼라 열 세개 벡터 한개지만 13개의 열을 표시하는거다
dense1 = Dense(30, name = 'hamsu1')(input1)          #summary로 모양을 보면 , 함수형 모델은 인풋레이어가 추가 출력된다.
dense2 = Dense(20, name = 'hamsu2')(dense1)
dense3 = Dense(10, name = 'hamsu3')(dense2)
output1 = Dense(1, name = 'hamsu4')(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()



#함수형모델 = 이 모델은 어디서 시작해서 어디서 끝이다 를 명시해줘야함
#함수형 모델은 인풋 레이어를 따로 명시해 줘야함.
#함수형 모델은 모델에 대한 정의를 먼저 하고 나중에 함수형 모델이라 정의함



# #3. 컴파일, 훈련

# model.compile(loss = 'mse', optimizer='adam',)
# model.fit(x_train,y_train,
#           epochs=10,
#           batch_size=32,
#           verbose=1,
#           )

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss :', loss)