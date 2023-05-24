# 일반적으로 컨볼루션이 느리다고 이야기 하지만 정작 연산량은 Dense가 더 많다.
# Flatten을 대체할 레이어 GlobalAveragePooling. 이전 레이어의 값에서 각 filter 마다 숫자를 평균 냄.

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
seed = 337
tf.random.set_seed(seed)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
model = Sequential()
model.add(Conv2D(64, (2,2),
                 padding = 'same', 
                 input_shape=(28,28,1)))
model.add(MaxPooling2D()) 
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(32, 2)) 
model.add(Flatten()) 
# model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

model.summary()

import time

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   mode='min',
                   verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience=5,
                        mode='auto',
                        factor=0.8,
                        verbose=1)

# 실행방법 : cmd에서 경로 이동 후  tensorboard --logdir=.
# 웹 켜고 : http://localhost:6006/
# 127.0.0.1:6006도 동일하다
tb = TensorBoard(log_dir='./_save/_tensorboard/_graph',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True,)

stt = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,epochs=50, batch_size = 128, validation_split=0.2, callbacks=[es, rlr, tb])
ett = time.time()


# Save the training history as a text file
#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])
print('걸린 시간 :', np.round(ett-stt,2))



########################## 시각화 ##########################
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 7))


# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()
