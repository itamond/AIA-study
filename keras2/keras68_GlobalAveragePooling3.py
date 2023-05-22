# 일반적으로 컨볼루션이 느리다고 이야기 하지만 정작 연산량은 Dense가 더 많다.
# 

from keras.datasets import mnist, cifar100, cifar10
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
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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
model.add(Dense(10, activation='softmax'))

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

