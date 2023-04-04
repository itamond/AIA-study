#https://www.kaggle.com/datasets/yapwh1208/dogs-breed-dataset

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split as tts
import time
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
  





#1. 데이터

x_train = np.load("d:/study_data/_save/dog's_breed/dog_breed_x_train150.npy")
y_train = np.load("d:/study_data/_save/dog's_breed/dog_breed_y_train150.npy")
x_test = np.load("d:/study_data/_save/dog's_breed/dog_breed_x_test150.npy")
y_test = np.load("d:/study_data/_save/dog's_breed/dog_breed_y_test150.npy")


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense,MaxPooling2D

# model = Sequential()
# model.add(Conv2D(256, (2,2), input_shape=(150, 150, 3),padding='same', activation=LeakyReLU(0.9)))
# model.add(MaxPooling2D())
# model.add(Conv2D(128, (2,2),padding='same', activation=LeakyReLU(0.9)))
# model.add(MaxPooling2D())
# model.add(Conv2D(64, (2,2), activation=LeakyReLU(0.9)))
# # model.add(MaxPooling2D())
# model.add(Conv2D(32, (2,2), activation=LeakyReLU(0.9)))
# # model.add(MaxPooling2D())
# model.add(Conv2D(16, (2,2), activation=LeakyReLU(0.9)))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(5,activation='softmax'))
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (3824, 150, 150, 3) (3824, 5)
# (206, 150, 150, 3) (206, 5)



model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

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

hist = model.fit(x_train, y_train, epochs=5000,   #x데이터 y데이터 배치사이즈가 한 데이터에 있을때 fit 하는 방법
                    # steps_per_epoch=32,    #전체데이터크기/batch = 160/5 = 32
                    validation_split=0.1,
                    shuffle=True,
                    batch_size = 16,
                    # validation_steps=24,    #발리데이터/batch = 120/5 = 24
                    callbacks=[es],
                    )


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('acc : ', acc[-1])
# print('val_acc : ', val_acc[-1])



ett = time.time()

# print('로드까지 걸린 시간 :', np.round(ett1-stt, 2))
# print('연산 걸린 시간 :', np.round(ett-stt, 2))

# from matplotlib import pyplot as plt

# plt.subplot(1,2,1)
# plt.plot(loss,label='loss')
# plt.plot(val_loss,label='val_loss')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(acc,label='acc')
# plt.plot(val_acc,label='val_acc')
# plt.legend()

# plt.show()


from sklearn.metrics import accuracy_score

result = model.evaluate(x_test,y_test)
print('result :', result)

pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test, pred)
print('acc:',acc)