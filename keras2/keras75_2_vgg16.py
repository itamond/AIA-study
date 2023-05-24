import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16


vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

vgg16.trainable = False  #vgg16의 가중치를 동결.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))


model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

#전이모델 또한 weight를 동결 시킬때, 동결 시키지 않을때 어느쪽이 성능이 좋다고 말할 수 없다.