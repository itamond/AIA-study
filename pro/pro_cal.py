#넘파이에서 불러와서 모델 구성
#성능 비교
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import Regularizer
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
#1. 데이터


# x_train = np.load('D:/number/_npy/pro_x.npy')
# y_train = np.load('D:/number/_npy/pro_y.npy')
x_pred = np.load('D:/number/_npy/pro_x_MMP.npy')
y_pred = np.load('D:/number/_npy/pro_y_MMP.npy')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(x_train.shape)
# print(y_train.shape) 
# print(x_test.shape) 
# print(y_test.shape)

# (121301, 28, 28, 1)
# (121301, 10)
# (200, 28, 28, 1)
# (200, 10)




model = Sequential()
model.add(Conv2D(32, (5,5), strides=1, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                 input_shape=(28, 28, 1)))
model.add(Conv2D(32, (5,5), strides=1, activation='relu',
                 use_bias=True,))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), strides= 1, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),))
model.add(Conv2D(64, (3,3), strides=1, use_bias=True))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, use_bias=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, use_bias=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(284, use_bias=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))



#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='acc',
                   mode = 'max',
                   patience=20,
                   verbose=1,
                   restore_best_weights=True,
                   )
vl= ReduceLROnPlateau(monitor='val_loss' , factor = 0.2, patience = 2)

hist = model.fit(x_train, y_train, epochs=30,  
                    validation_split=0.1,
                    shuffle=True,
                    batch_size = 512,
                    callbacks=[es,vl],
                    )


result = model.evaluate(x_test,y_test)
print('result :', result)
print(x_pred)
pred = np.argmax(model.predict(x_pred), axis=1)
print(y_pred)
y_pred = np.argmax(y_pred,axis=1)
print(y_pred)
acc = accuracy_score(y_pred, pred)
print('pred :', pred)
print('acc:',acc)

# [6 1 2 7 6 2 4 2 2 2 3 9 7 2 6 8 3 0 4 2 8 6 2 7 8 9 4 7 0 8 4 4 3 6 0 3 9
#  3 6 7 4 6 2 1 3 3 2 2 4 3 3 1 3 2 1 7 7 0 7 4 7 8 0 2 4 3 6 3 4 6 4 4 9 1
#  7 9 7 4 6 6 7 2 2 7 1 3 7 4 4 7 6 3 4 3 3 1 6 7 3 1 7 3 2 6 3 6 2 9 9 8 6
#  2 7 6 2 3 9 2 9 5 8 3 7 7 9 0 0 7 3 6 2 1 0 2 6 4 3 3 0 3 7 4 3 9 2 4 2 3
#  3 0 3 3 9 6 2 9 3 6 3 3 2 8 9 7 7 4 5 2 4 3 1 3 4 3 8 7 2 4 2 1 2 6 2 4 7
#  7 7 0 3 9 3 8 4 3 7 1 1 1 3 3]