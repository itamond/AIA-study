import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar100, cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

base_model = VGG16(weights = 'imagenet', include_top = False, input_shape=(32, 32, 3))
# print(base_model.output)
# KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 512), dtype=tf.float32, name=None), name='block5_pool/MaxPool:0', description="created by layer 'block5_pool'")
x = base_model.output #base_model의 최종 레이어 
x = GlobalAveragePooling2D()(x)
output1 = Dense(10, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = output1)

# model.summary()

es = EarlyStopping(monitor = 'val_loss',
                   patience = 15, mode = 'min',
                   verbose = 1)

rlr = ReduceLROnPlateau(monitor = 'val_loss',
                        patience = 3,
                        mode='auto',
                        verbose = 1,
                        factor=0.5)

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 512, validation_split=0.2, batch_size = 512, callbacks=[es, rlr])

y_pred = np.argmax(model.predict(x_test), axis = 1)
y_test = np.argmax(y_test, axis = 1)

acc = accuracy_score(y_test, y_pred)
print('acc: ', acc)