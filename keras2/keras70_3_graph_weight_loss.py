#loss와 weight의 관계 그리기
# 
# 
# # 일반적으로 컨볼루션이 느리다고 이야기 하지만 정작 연산량은 Dense가 더 많다.
# Flatten을 대체할 레이어 GlobalAveragePooling. 이전 레이어의 값에서 각 filter 마다 숫자를 평균 냄.

# epoch와 loss / acc 그래프
# 훈련하지 말고 70_1에서 세이브한 것 불러와서 그리기

from keras.datasets import mnist
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
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
model = load_model('./_save/keras70_1_mnist_graph.h5')
# model.add(Conv2D(64, (2,2),
#                  padding = 'same', 
#                  input_shape=(28,28,1)))
# model.add(MaxPooling2D()) 
# model.add(Conv2D(filters=64, kernel_size=(2,2), padding='valid', activation='relu'))
# model.add(Conv2D(32, 2)) 
# model.add(Flatten()) 
# # model.add(GlobalAveragePooling2D())
# model.add(Dense(10, activation='softmax'))

# model.summary()

import time

#3. 컴파일, 훈련

# stt = time.time()
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# hist = model.fit(x_train, y_train,epochs=50, batch_size = 128, validation_split=0.2)
# ett = time.time()

#4. 평가, 예측
# results = model.evaluate(x_test, y_test)
# print('loss :', results[0])
# print('acc :', results[1])
# print('걸린 시간 :', np.round(ett-stt,2))

# model.save('./_save/keras70_1_mnist_graph.h5')


model_weights = model.get_weights()

weight_values = np.array(model_weights)

import pickle

with open('./_save/keras70_1_mnist_grape.pkl', 'rb') as f:
    hist = pickle.load(f)
########################## 시각화 ##########################



import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist['loss'], marker='.', c='red', label='loss')
plt.plot(weight_values, marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2)
plt.plot(hist['acc'], marker='.', c='red', label='acc')
plt.plot(hist['val_acc'], marker='.', c='blue', label='acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()



