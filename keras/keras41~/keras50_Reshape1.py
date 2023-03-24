#모델에서 리쉐이프 하는 법


from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D, Reshape, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
#1. 데이터




(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
     

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# (60000, 28, 28) (60000,)
# (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))



x_train = x_train.reshape(60000, 28, 28, 1)/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.         #reshape와 scaling 동시에 하기.
print(x_train.shape, y_train.shape)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


#2. 모델
model =Sequential()
model.add(Conv2D(filters=64, kernel_size = (3,3),
                 padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3)))
model.add(Conv2D(10,3))    #커널 사이즈 하나면 써도 됨.
model.add(MaxPooling2D())
model.add(Flatten())    #(N, 250)
model.add(Reshape(target_shape=(25,10)))     #리쉐이프는 연산량 없다.
model.add(Conv1D(10, 3, padding='same'))
model.add(LSTM(250))
model.add(Reshape(target_shape=(50, 5, 1)))
model.add(Conv2D(32,(3,3), padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 28, 28, 64)        640

#  max_pooling2d (MaxPooling2D  (None, 14, 14, 64)       0
#  )

#  conv2d_1 (Conv2D)           (None, 12, 12, 10)        5770

#  conv2d_2 (Conv2D)           (None, 10, 10, 10)        910

#  max_pooling2d_1 (MaxPooling  (None, 5, 5, 10)         0
#  2D)

#  flatten (Flatten)           (None, 250)               0

#  reshape (Reshape)           (None, 25, 10)            0

#  conv1d (Conv1D)             (None, 25, 10)            310

#  lstm (LSTM)                 (None, 250)               261000

#  reshape_1 (Reshape)         (None, 50, 5, 1)          0

#  conv2d_3 (Conv2D)           (None, 50, 5, 32)         320

#  flatten_1 (Flatten)         (None, 8000)              0

#  dense (Dense)               (None, 10)                80010

# =================================================================
# Total params: 348,960
# Trainable params: 348,960
# Non-trainable params: 0
# _________________________________________________________________