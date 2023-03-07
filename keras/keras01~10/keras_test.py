#1. 데이터

import numpy as np
x= np.array([[1,2],[3,4]])
y= np.array([4,5])


#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#케라스 = 텐서플로가 어려워서 나온 api
model=Sequential()
model.add(Dense(3,input_dim=2))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=10, batch_size=1)

#4. 평가, 예측

loss=model.evaluate(x,y)
print("loss :", loss)

result=model.predict([[8,9]])
print("[[8]]의 예측값 :", result)