#넘파이에서 불러와서 모델 구성
#성능 비교

import numpy as np
from keras.preprocessing.image import ImageDataGenerator #이미지데이터를 수치화
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score



#1. 데이터


x_train = np.load('d:/study_data/_save/_npy/58_horse_human_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/58_horse_human_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/58_horse_human_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/58_horse_human_test_y.npy')

#print(x_train)
print(x_train.shape)
print(y_train.shape) 
print(x_test.shape) 
print(y_test.shape) 

#2. 모델



model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc',
                   mode = 'max',
                   patience=30,
                   verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit(x_train, y_train, epochs=5000,  
                    validation_split=0.1,
                    shuffle=True,
                    batch_size = 64,
                    callbacks=[es],
                    )


result = model.evaluate(x_test,y_test)
print('result :', result)

pred = np.round(model.predict(x_test))
y_test = np.round(y_test)
acc = accuracy_score(y_test, pred)
print('acc:',acc)

#acc: 0.7330097087378641