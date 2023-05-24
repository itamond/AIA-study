import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator


input1 = Input(shape=(32, 32, 3))
vgg16 = VGG16(include_top = False, weights = 'imagenet')(input1)
# gap1 = GlobalAveragePooling2D()(vgg16)
flt1 = Flatten()(vgg16)
hidden1 = Dense(100)(flt1)
output1 = Dense(10, activation='softmax')(hidden1)

model = Model(inputs = input1, outputs = output1)

model.summary()