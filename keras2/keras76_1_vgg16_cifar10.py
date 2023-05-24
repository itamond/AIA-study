import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train_resized = tf.image.resize(x_train, (224, 224))
# x_test_resized = tf.image.resize(x_test, (224, 224))
# datagen = ImageDataGenerator()

# datagen.fit(x_train_resized)
# x_train_augmented = datagen.flow(x_train_resized, y_train, batch_size=len(x_train)).next()[0]

# datagen.fit(x_test_resized)
# x_test_augmented = datagen.flow(x_test_resized, y_test, batch_size=len(x_test)).next()[0]

vgg16_f = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# vgg16_t = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

for i in range(5):
    if i < 1:
        model = Sequential()
        model.add(Dense(32, input_shape=(32, 32, 3)))
    elif 1 <= i < 3:
        model = Sequential()
        model.add(vgg16_f)
    elif 3 <= i:
        model = Sequential()
        vgg16_f.trainable = False
        model.add(vgg16_f)

    if i == 0 or i == 1 or i == 3:
        model.add(Flatten())
    elif i == 2 or i == 4:
        model.add(GlobalAveragePooling2D())
    
    model.add(Dense(32))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=128)
    result = model.evaluate(x_test, y_test)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))
    
    print(f'{i+1} result : {result}\nacc : {acc}')
    
    
    
#0번째-> vgg를 안쓴 기본 덴스 모델
#1번째-> trainable True에 Flatten
#2번재-> trainable True에 GAP
#3번째-> trainable False에 Flatten
#4번째-> trainable False에 GAP

# Trainable:True   model False   VGG False
#      30              30           30
#      30              0         layer갯수*2