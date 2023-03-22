import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터



#2. 모델구성
model = Sequential()                       #[batch(행의 크기 데이터의 갯수), timesteps(몇개씩 잘라서 쓸건지), feature(몇개 씩 훈련할건지)]
# model.add(LSTM(10, input_shape=(5, 1)))                        #input_shape = 행 빼고 나머지
# model.add(LSTM(10, input_length=5, input_dim=1))              #인풋 쉐이프가 먹히지 않는 상황이 있다. 그럴때 인풋 렝스와 딤을 사용해야할 때가 있다.
model.add(LSTM(10,  input_dim=1, input_length=5))              # 앞뒤 바꾸기도 가능, 가독성 떨어짐 비추 
                                                            #따라서 [batch , input_length, input_dim] 이라고 부를 수도 있다. 단지 표현의 차이이다.
model.add(Dense(7, activation='relu'))

model.add(Dense(1))
model.summary()


