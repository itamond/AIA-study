#이미지 데이터를 수치화 하는 과정
#이미지 데이터를 증폭하는 과정 및 옵션

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping



#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,              #MinMax스케일링(정규화) 하겠다는 의미, . 을 붙힌 이유는 부동소수점으로 연산해라 라는 뜻
    horizontal_flip=True,        #가로 뒤집기
    vertical_flip=True,          #수직 뒤집기
    width_shift_range=0.1,       #10%만큼을 좌우로 움직일 수 있다는 뜻
    height_shift_range=0.1,      #상하로 10% 움직일 수 있다는 뜻
    rotation_range=5,            #돌릴 수 있는 범위
    zoom_range=1.2,              #20%만큼 확대하겠다는 뜻
    shear_range=0.7,             #찌그러트릴 수 있는 범위
    fill_mode='nearest',         #이미지를 움직일 때, 움직여서 없어진 범위에 근처의 값을 입력해주는 기능
)                                #숫자 6하고 9같은 반전하면 데이터가 꼬이는 경우도 있다. 이럴 경우 옵션 조절해야함

test_datagen = ImageDataGenerator(
    rescale=1./255,            
    # horizontal_flip=True,       #테스트데이터는 평가하는 데이터이기때문에 데이터를 증폭한다는건 결과를 조작하는것이다. 때문에 스케일 제외한 옵션들 삭제 
    # vertical_flip=True,         #통상적으로 테스트 데이터는 증폭하지 않는다
    # width_shift_range=0.1,      
    # height_shift_range=0.1,    
    # rotation_range=5,         
    # zoom_range=1.2,           
    # shear_range=0.7,        
    # fill_mode='nearest',
)   


xy_train =train_datagen.flow_from_directory(                     #폴더에서 가져올거야~    
    'd:/study_data/_data/brain/train/',                #이미지제너레이터는 폴더별로 라벨값 부여. 때문에 분류 폴더 이전 상위폴더까지만 설정해도됨
    target_size=(100, 100),                            #이미지 데이터를 200x200으로 확대 혹은 축소해라. 사이즈를 동일하게 만들어준다.
    batch_size=200,                                      #5장씩 잘라라
    class_mode='binary',                               #0과 1을 찾는 mode, int형 수치화해서 만들어줌 
    # color_mode='rgba',
    color_mode='grayscale',
    shuffle=True,
)   #Found 160 images belonging to 2 classes.   0과 1의 클래스로 분류되었다.        #x=160, 200, 200, 1 로 변환 됐음  y=160,

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(100, 100),
    batch_size=5,                                      #전체 데이터를 배치로 잡아도 된다.
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)   #Found 120 images belonging to 2 classes.   0과 1의 클래스로 분류되었다.        #x=120, 200, 200, 1 로 변환 됐음  y=120,

# np.unique = pd.value_counts
# print(xy_train)        
#<keras.preprocessing.image.DirectoryIterator object at 0x000002E1DFC72790>
#Iterator 반복자. 반복자의 대표 LIST

# print(xy_train[0])
# #pirnt(xy_train.shape)
# print(len(xy_train))     #32
# print(len(xy_train[0]))  #2     #xy_train의 첫번째 데이터는 배치5 사이즈의 x와 y  한묵음이다.

# print(xy_train[0][0]) # batch_size개의 x가 들어가있다.
# print(xy_train[0][1]) # batch_size개의 y가 들어가있다.
print(xy_train[0][0].shape)   #(batch_size, 200, 200, 1) shape가 먹힌다는건 넘파이라는것
print(xy_train[0][1].shape)   #(batch_size,)
# #x와 y가 합쳐진 이터레이터 형태의 데이터이다.

print("=========================================================")
print(type(xy_train))      #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))   #<class 'tuple'>
print(type(xy_train[0][0]))#<class 'numpy.ndarray'>
print(type(xy_train[0][1]))#<class 'numpy.ndarray'>

#현재 x는 (5,200,200,1) 짜리 데이터가 32덩어리


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
model.summary()
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(xy_train[:][0], xy_train[:][1],
#           epochs=10,
#           )   #에러

es = EarlyStopping(monitor='val_acc',
                   mode = 'max',
                   patience=10,
                   verbose=1,
                   restore_best_weights=True,
                   )

hist =model.fit(xy_train[0][0], xy_train[0][1],
          epochs=10,
          validation_data=(xy_test[0][0], xy_test[0][1])
          )   #전체 데이터를 배치로 잡으면 가능

# hist = model.fit(xy_train, epochs=3000,   #x데이터 y데이터 배치사이즈가 한 데이터에 있을때 fit 하는 방법
#                     steps_per_epoch=32,    #전체데이터크기/batch = 160/5 = 32
#                     validation_data=xy_test,
#                     validation_steps=24,    #발리데이터/batch = 120/5 = 24
#                     )


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