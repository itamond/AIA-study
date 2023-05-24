import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


input1 = Input(shape=(32, 32, 3))
vgg16 = VGG16(weights='imagenet', include_top=False)(input1)

flat1 = Flatten()(vgg16)

dense1 = Dense(128,activation='relu')(flat1)
dense2 = Dense(64,activation='relu')(dense1)
output1 = Dense(10,activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)   
# model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 50, validation_split=0.2, batch_size = 512)

y_pred = np.argmax(model.predict(x_test), axis = 1)
y_test = np.argmax(y_test, axis = 1)

acc = accuracy_score(y_test, y_pred)
print('acc: ', acc)