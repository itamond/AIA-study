import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten


#1. 데이터

#2. 모델
model=Sequential()
# model.add(LSTM(10, input_shape=(3,1))) # 토탈파람스 : 541
model.add(Conv1D(10, 2, input_shape=(3,1)))    # 필터갯수, 커널, 쉐이프   토탈파람스 : 141
model.add(Conv1D(10, 2)) #토탈파람스 : 301         
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))
model.summary()

#Conv1D의 파라미터 계산은 Conv2D와 같다.
#Conv1D은 LSTM의 값을 다음으로 넘기는것과 달리 특성을 추출한다
#Conv1D는 3차원 입력 3차원 출력