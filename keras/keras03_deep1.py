# 1. 데이터
import numpy as np
x = np.array([1, 2, 3]) # array= 배열 []=묶음
y = np.array([1, 2, 3])

# 2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential # Sequential = 순차적 모델
from tensorflow.keras.layers import Dense # 레이어 = 층, 노드(뉴런)의 집합 , Dense 레이어는 다음 층의 노드가 이전 계층의 노드값을 모두 입력 받음

model = Sequential()
model.add(Dense(3, input_dim=1)) #인풋 레이어= 1 dim=차원,디멘션(1,2,3이라는 한 덩어리)
model.add(Dense(4)) #위에 있는 3이 인풋이기 때문에 input_dim 은 명시하지 않아도 된다
model.add(Dense(7))
model.add(Dense(5)) 
model.add(Dense(3)) #여기까지 Hidden 레이어
model.add(Dense(1)) #아웃풋 레이어

# 3. 컴파일, 훈련    컴파일= 기계어로 바꾸는것, 내가 던져준 코드를 알아 듣도록 해라.
model.compile(loss='mse', optimizer='adam') #loss를 mse로 계산하라 mse=오차(error)를 제곱한 값의 평균
model.fit(x, y, epochs=100)  #fit에 x와 y 데이터로 100번 훈련시켜라

# loss: 0.0023