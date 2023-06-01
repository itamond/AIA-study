# [실습] keras56_4 남자 여자 noise 넣기
# predict : 기미 주근깨 제거
# 5개 사진 출력 / 원본, 노이즈, 아웃풋

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from tensorflow.keras.applications import ResNet101V2
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAvgPool2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import time

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                validation_split=0.2)


test_datagen = ImageDataGenerator(rescale=1./255)

xy = train_datagen.flow_from_directory(
    'C:/AIA/men_women',
    target_size = (150, 150),
    batch_size = 500,
    class_mode = 'binary',
    shuffle = True)

x = xy[0][0]
y = xy[0][1]


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle= True, random_state= 337)


x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


from keras.models import Sequential, Model
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D



def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size,input_shape=(150, 150, 3),kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
    model.add(Conv2D(32,(2,2),padding='same',activation='relu'))
    model.add(UpSampling2D())
    
    model.add(Conv2D(hidden_layer_size,(2,2),padding='same',activation='relu'))
    model.add(Conv2D(3,(2,2),padding='same',activation='sigmoid'))
    return model


model = autoencoder(hidden_layer_size=154) # PCA의 95% 성능

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train_noised, x_train, epochs=2)
output = model.predict(x_test_noised)


import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), 
      (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20, 7))
    
# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150, 3))
    if i ==0:
        ax.set_ylabel("Input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    
# 노이즈가 들어간 이미지를 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3))
    if i ==0:
        ax.set_ylabel("Noise", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(150, 150, 3))
    if i ==0:
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    
plt.tight_layout()
plt.show()