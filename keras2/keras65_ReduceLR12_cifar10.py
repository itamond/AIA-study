#ReduceLRonPlateau : 개선이 없으면 Learningrate 반으로 줄인다.


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split as tts
from keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import accuracy_score

#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)
# (50000, 32, 32, 3) (50000, 1)

x_train = x_train.reshape(50000, 32*32*3)/255.
x_test = x_test.reshape(10000, 32*32*3)/255.         #reshape와 scaling 동시에 하기.

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Conv2D(64, 
                 (3,3),
                 padding='same',
                 input_shape = (32, 32, 3),
                 activation='relu'))
model.add(Conv2D(64,
                 (3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D())   
model.add(Conv2D(32,
                 (3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D())   
model.add(Conv2D(16,
                 (3,3),
                 padding='same',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate,)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor = 'val_loss',
                   patience = 10, mode = 'min',
                   verbose = 1)

rlr = ReduceLROnPlateau(monitor = 'val_loss',
                        patience = 2,
                        mode='auto',
                        verbose = 1,
                        factor=0.5)

model.fit(x_train, y_train,
          epochs = 50, 
          batch_size = 512, 
          validation_split=0.2, 
          verbose=1, 
          callbacks=[es, rlr])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test,axis=1)




acc = accuracy_score(y_pred, y_test)
print(acc)

# 0.584
