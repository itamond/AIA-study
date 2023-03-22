from keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

import matplotlib.pyplot as plt

# plt.imshow(x_train[32])           #그림 그려주는 함수
plt.imshow(x_train[39083])
plt.show()
