#이미지 데이터를 수치화 하는 과정
#이미지 데이터를 증폭하는 과정 및 옵션
#이미지 데이터를 매번 수치화 하면 시간이 너무 오래 걸린다.
#넘파이 형태로 수치화 한것을 저장해두면 시간을 단축할 수 있다.
#저장한 데이터를 불러오는 방법


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping



#1. 데이터

path = 'd:/study_data/_save/_npy/'
# np.save(path + 'keras55_1_x_train.npy', arr=xy_train[0][0])          #수치화된 데이터를 np형태로 저장
# np.save(path + 'keras55_1_x_test.npy', arr=xy_train[0][0])    
# np.save(path + 'keras55_1_y_train.npy', arr=xy_train[0][1])    
# np.save(path + 'keras55_1_y_test.npy', arr=xy_train[0][1])    

x_train = np.load(path+'keras55_1_x_train.npy')
x_test = np.load(path+'keras55_1_x_test.npy')
y_train = np.load(path+'keras55_1_y_train.npy')
y_test = np.load(path+'keras55_1_y_test.npy')
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)       #(160, 100, 100, 1)
# (160, 100, 100, 1) (120, 100, 100, 1) (160,) (120,)

# 현재 x는 (5,200,200,1) 짜리 데이터가 32덩어리


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense,MaxPooling2D

model = Sequential()
model.add(Conv2D(256, (2,2), input_shape=(100, 100, 1),padding='same', activation=LeakyReLU(0.9)))
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
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
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
                    steps_per_epoch=32,    #전체데이터크기/batch = 160/5 = 32
                    validation_data=(x_test,y_test),
                    validation_steps=24,    #발리데이터/batch = 120/5 = 24
                    callbacks=[es]
                    )


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print(acc)
print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_ac : ', val_acc[-1])


#1. 그림그려              subplot()
# 하나는 로스 발로스
# 하나는 애큐 발애큐

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