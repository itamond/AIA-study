from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
(x_train,y_train),(x_test,y_test) = mnist.load_data()   #파이썬 기초 문법. _ 언더바 입력시 메모리에 할당하지 않음

#28x28의 이미지의 경우 784개의 컬런이라고 볼 수도 있다. (60000,784)
#이미지의 데이터는 쓸모없는 0의 데이터가 많다. 이를 PCA를 통해 압축시킬 수 있다.
#EVR을 통해 몇개의 컬런을 지워야 성능이 좋을지 확인한다.

# x = np.concatenate((x_train,x_test),axis=0) #train과 test를 붙히는 방법
# x = np.append(x_train,x_test, axis=0)  #둘 다 가능하다.
# print(x.shape)
#(70000, 28, 28)



###########실습############
# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95 몇개?
# 0.99 몇개?
# 0.999 몇개?
# 1.0 몇개?
x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)



# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)
# print(cumsum)


# print(np.argmax(cumsum >= 0.95) + 1)    #154 
# print(np.argmax(cumsum >= 0.99) + 1)    #331 
# print(np.argmax(cumsum >= 0.999) + 1)    #486 
# print(np.argmax(cumsum >= 1.0) + 1)    #713 숫자는 0부터 시작하므로 1 더해준다.

################실습#################

# acc
#1. 나의 최고 CNN : 0.0000
#2. 나의 최고 DNN : 0.0000
#3. PCA 0.95 :
#4. PCA 0.99 :
#5. PCA 0.999 :
#6. PCA 1 :


import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import datetime
import time


scaler= MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델구성


es = EarlyStopping(monitor = 'val_acc',
                   patience = 50,
                   verbose = 1,
                   restore_best_weights=True,
                   mode = 'auto'
                   )

# pca = PCA(n_components=784)  # [154, 331, 486, 713]
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)



pca_num = [154, 331, 486, 713]
print('1. 나의 최고 CNN : 0.9881')
print('2. 나의 최고 DNN : 0.9685')
for i,v in enumerate(pca_num) :
    pca = PCA(n_components=v)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    model =Sequential()
    model.add(Dense(64, input_shape=(v,)))
    model.add(Dropout(0.5))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              )
    model.fit(x_train,y_train,
              epochs=10,
              validation_split=0.2,
              verbose=0,
              batch_size=32,
              callbacks=[es])
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict, axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = np.round(accuracy_score(y_predict, y_true),4)
    print(i+3,'PCA',v,':',acc)

end_time = time.time()
# #4. 평가, 예측

# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
# y_true = np.argmax(y_test, axis=1)
# acc = accuracy_score(y_predict, y_true)

# loss : 0.11206462234258652
# acc : 0.968500018119812
# acc : 0.9685
# 걸린시간 : 1495.55