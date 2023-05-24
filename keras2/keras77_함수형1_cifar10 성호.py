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
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

vgg16_f = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

result_list = []
test_acc_list = []

for i in range(5):
    if i < 1:
        input1 = Input(shape=(32, 32, 3))
        dense1 = Dense(100)(input1)
    elif 1 <= i < 3:
        input1 = Input(shape=(32, 32, 3))
        dense1 = vgg16_f(input1)
    elif 3 <= i:
        input1 = Input(shape=(32, 32, 3))
        vgg16_f.trainable = False
        dense1 = vgg16_f(input1)

    if i == 0 or i == 1 or i == 3:
        flat = Flatten()(dense1)
    
    elif i == 2 or i == 4:
        flat = GlobalAveragePooling2D()(dense1)
    
    dense2 = Dense(100)(flat) 
    output = Dense(10, activation='softmax')(dense2)
    
    model = Model(inputs=input1, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    result = model.evaluate(x_test, y_test)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))
    
    print(f'{i+1} result : {result}\nacc : {acc}')
    result_list.append(result[0])
    test_acc_list.append(acc)

print(f'{result_list}\n{test_acc_list}')