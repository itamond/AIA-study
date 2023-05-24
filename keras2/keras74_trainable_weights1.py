import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(337)

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)
#레이어 상에서 kernel이라는 것은 일반적으로 weight를 의미한다.
print('======================================================================================')
print(model.trainable_weights)

print(len(model.weights)) #6 커널과 바이어스의 갯수
print(len(model.trainable_weights)) #6

model.trainable = False #

print(len(model.weights)) 
print(len(model.trainable_weights))

print('======================================================================================')
print(model.weights)
print('======================================================================================')
print(model.trainable_weights)

model.summary()