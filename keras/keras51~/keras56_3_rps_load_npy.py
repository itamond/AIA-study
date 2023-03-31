

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import time

#1. 데이터

path = 'd:/study_data/_save/rps/'

stt = time.time()
x = np.load(path+'keras56_5_x_train.npy')
# x_test = np.load(path+'keras55_1_x_test.npy')
y = np.load(path+'keras56_5_y_train.npy')
# y_test = np.load(path+'keras55_1_y_test.npy')
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)       #(160, 100, 100, 1)
# (160, 100, 100, 1) (120, 100, 100, 1) (160,) (120,)

# 현재 x는 (5,200,200,1) 짜리 데이터가 32덩어리
ett1 = time.time()



x_train, x_test, y_train, y_test =  tts(x, y,
                                        train_size=0.7,
                                        random_state=32,
                                        # stratify=y,
                                        )

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense,MaxPooling2D

model = Sequential()
model.add(Conv2D(256, (2,2), input_shape=(150, 150, 3),padding='same', activation=LeakyReLU(0.9)))
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), activation=LeakyReLU(0.9)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), activation=LeakyReLU(0.9)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])



# model.fit(xy_train[:][0], xy_train[:][1],
#           epochs=10,
#           )   #에러



es = EarlyStopping(monitor='val_acc',
                   mode = 'max',
                   patience=30,
                   verbose=1,
                   restore_best_weights=True,
                   )

# model.fit(xy_train[0][0], xy_train[0][1],
#           epochs=10,
#           )   #전체 데이터를 배치로 잡으면 가능
# hist = model.fit_generator(xy_train, epochs=3000,   #x데이터 y데이터 배치사이즈가 한 데이터에 있을때 fit 하는 방법
#                     steps_per_epoch=32,    #전체데이터크기/batch = 160/5 = 32
#                     validation_data=xy_test,
#                     validation_steps=24,    #발리데이터/batch = 120/5 = 24
#                     )

hist = model.fit(x_train, y_train, epochs=3000,   #x데이터 y데이터 배치사이즈가 한 데이터에 있을때 fit 하는 방법
                    # steps_per_epoch=32,    #전체데이터크기/batch = 160/5 = 32
                    validation_split=0.2,
                    batch_size = 8,
                    # validation_steps=24,    #발리데이터/batch = 120/5 = 24
                    callbacks=[es],
                    )


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']


ett = time.time()

print('로드까지 걸린 시간 :', np.round(ett1-stt, 2))
print('연산 걸린 시간 :', np.round(ett-stt, 2))

from matplotlib import pyplot as plt

plt.subplot(1,2,1)
plt.plot(loss,label='loss')
plt.plot(val_loss,label='val_loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(acc,label='acc')
plt.plot(val_acc,label='val_acc')
plt.legend()

plt.show()

pred = np.argmax(model.predict(x_test),axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test, pred)
print('acc :', acc)