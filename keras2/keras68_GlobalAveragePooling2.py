# 일반적으로 컨볼루션이 느리다고 이야기 하지만 정작 연산량은 Dense가 더 많다.
# 

from keras.datasets import mnist, cifar100
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
import tensorflow as tf
seed = 337
tf.random.set_seed(seed)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(np.unique(y_train, return_counts=True))

print(x_train.shape)

# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
model = Sequential()
model.add(Conv2D(64, (2,2), 
                 padding = 'same', 
                 input_shape=(32,32,3)))
model.add(MaxPooling2D()) 
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(32, 2)) 
# model.add(Flatten()) 
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))

model.summary()

import time

#3. 컴파일, 훈련

stt = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,epochs=100, batch_size = 128, validation_split=0.368)
ett = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])
print('걸린 시간 :', np.round(ett-stt,2))


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        320
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 64)        16448
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 12, 12, 32)        8224
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 32)                0 연산을 1%이하 수준으로 줄여줌
# _________________________________________________________________
# dense (Dense)                (None, 10)                330
# =================================================================
# Total params: 25,322
# Trainable params: 25,322
# Non-trainable params: 0
# _________________________________________________________________



"""
Flatten 결과
loss : 0.20656675100326538
acc : 0.9821000099182129
걸린 시간 : 30.59
"""


"""
Global_average_pooling 결과
loss : 0.15325693786144257
acc : 0.9509999752044678
걸린 시간 : 29.88
"""

"""
loss : 44.45093536376953
acc : 0.15929999947547913
걸린 시간 : 70.09
"""

"""
loss : 2.6420485973358154
acc : 0.3504999876022339
걸린 시간 : 67.83
"""