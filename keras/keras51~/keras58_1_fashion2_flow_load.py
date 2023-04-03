#넘파이에서 불러와서 모델 구성
#성능 비교

import numpy as np
from keras.preprocessing.image import ImageDataGenerator #이미지데이터를 수치화
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score



#1. 데이터


x_train = np.load('d:/study_data/_save/_npy/58_fashion_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/58_fashion_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/58_fashion_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/58_fashion_test_y.npy')

#print(x_train)
print(x_train.shape) #(100000, 28, 28, 1)
print(y_train.shape) #(100000,)
print(x_test.shape)  #(10000, 28, 28, 1)
print(y_test.shape)  #(10000,)

#2. 모델



model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc',
                   mode = 'max',
                   patience=30,
                   verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit(x_train, y_train, epochs=5000,  
                    validation_split=0.1,
                    shuffle=True,
                    batch_size = 256,
                    callbacks=[es],
                    )


result = model.evaluate(x_test,y_test)
print('result :', result)

pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test, pred)
print('acc:',acc)




#acc: 0.8814