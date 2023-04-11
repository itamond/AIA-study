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


x_train = np.load('D:/number/_npy/pro_x_cb.npy')
y_train = np.load('D:/number/_npy/pro_y_cb.npy')
x_pred = np.load('D:/number/_npy/pro_x_MMP.npy')
y_pred = np.load('D:/number/_npy/pro_y_MMP.npy')
# (190481, 28, 28, 1) (190481, 10)




# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
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
                   patience=5,
                   verbose=1,
                   restore_best_weights=True,
                   )
vl= ReduceLROnPlateau(monitor='val_loss' , factor = 0.2, patience = 2)

hist = model.fit(x_train, y_train, epochs=30,  
                    validation_split=0.1,
                    shuffle=True,
                    batch_size = 128,
                    callbacks=[es,vl],
                    )

model.save_weights('D:/number/h5/pro_cal.h5')

# result = model.evaluate(x_test,y_test)
# print('result :', result)
print(x_pred)
pred = np.argmax(model.predict(x_pred), axis=1)
print(y_pred)
y_pred = np.argmax(y_pred,axis=1)
print(y_pred)
acc = accuracy_score(y_pred, pred)
print('pred :', pred)
print('acc:',acc)


#Mnist로 train, gan Mnist,Mnist predict
# [0 0 1 1 1 1 1 1 2 2 3 3 4 4 4 4 4 4 5 5 6 6 7 8 8 9 9 9]
# [0 0 1 1 1 1 1 1 2 2 3 3 1 4 4 6 4 4 5 5 6 6 7 8 8 9 3 9]
# acc: 0.8928571428571429


# [0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5 5 6
#  6 6 6 6 6 7 7 7 7 7 7 8 8 8 8 8 8 9 9 9 9 9 9]
# [0 0 0 2 0 0 1 1 1 4 1 1 2 2 2 4 2 2 3 3 3 9 2 4 4 4 4 4 4 4 5 5 5 5 5 5 6
#  6 5 6 6 9 7 7 1 2 1 2 8 8 3 9 3 8 9 9 1 8 1 1]
# acc: 0.6833333333333333

# [0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5 5 6
#  6 6 6 6 6 7 7 7 7 7 7 8 8 8 8 8 8 9 9 9 9 9 9]
# [0 0 0 2 0 0 1 1 1 4 1 1 2 2 4 4 2 4 3 3 3 9 3 3 4 4 6 4 4 4 5 5 4 5 5 5 6
#  6 4 6 6 5 7 7 1 1 1 2 8 8 8 9 3 8 9 9 1 1 1 1]
# acc: 0.6666666666666666