#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential #텐서플로 안에 케라스 안에 모델스 안에 시퀀셜을 가져와라
from tensorflow.keras.layers import Dense #텐서플로 안에 케라스 안에 레이어스 안에 덴스를 가져와라

model = Sequential() #모델은 시퀀셜 모델이다
model.add(Dense(1, input_dim=1)) # 노드의 갯수

#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam') #로스는 mse로, 옵티마이저->최적화는 아담으로 써라 
model.fit(x, y, epochs=100) #에폭스는 훈련 횟수, fit= 훈련을 시키다. epochs는 시행 횟수

#훈련량이 많다고 해서 값이 무조건 좋아지는 것은 아니다.

# loss: 0.730