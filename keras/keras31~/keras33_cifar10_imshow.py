from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)

# x_train = x_train.reshape(50000, 32*32*3)
# x_test = x_test.reshape(10000, 32*32*3)


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#****************x_train = x_train/255.     - 콤마는 파이썬의 부동 소수점 연산이라는 표시
#****************x_test = x_test/255.


x_train = x_train.reshape(50000, 32*32*3)/255.
x_test = x_test.reshape(10000, 32*32*3)/255.         #reshape와 scaling 동시에 하기.

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

import matplotlib.pyplot as plt

# plt.imshow(x_train[32])           #그림 그려주는 함수
plt.imshow(x_train)
plt.show()
