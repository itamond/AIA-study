import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
#다중분류, 

#1. 데이터

# datasets=fetch_covtype()
a = fetch_covtype()
x = a.data
y = a.target
#7 클래스
# print(a.DESCR) 
#(581012, 54)
#(581012,)

y = to_categorical(y)
y = np.delete(y, 0, axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    stratify=y,
    random_state=388,
)


# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)


#2. 모델 구성

model=Sequential()
model.add(Dense(108, input_dim=54))
model.add(Dense(216, activation='relu'))
model.add(Dense(108, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(7, activation='softmax'))

# model.summary()

# Total params: 59,191
# Trainable params: 59,191

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy',                 #sparse_categorical_crossentropy
              optimizer = 'adam',                              #원핫을 안해도 된다.
              metrics= ['acc'],
              )
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=20,
                   verbose=1,
                   )
import time
start_time = time.time()  #현 시점에서의 시간값을 start_time에 반환

hist = model.fit(x_train, y_train,
                 epochs = 10,
                 batch_size = 5000,
                 callbacks=[es],
                 verbose=1,
                 validation_split=0.2,
                 )

end_time = time.time()     #현 시점에서의 시간값을 end_time 에 반환

print('걸린시간 : ', round(end_time - start_time,2))    # round의 2는 소수 둘째까지 반환하라는것


#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_argm = np.argmax(y_test, axis=1)


acc = accuracy_score(y_argm, y_pred)
print('acc :', acc)


#배치 10k , 에포 100, 페이션스 10, 발리데이션 스플릿 0.2 랜덤= 388
#result : [0.6699721217155457, 0.7153860330581665]
# acc : 0.7153860055248144



#배치 5k , 에포 100, 페이션스 10, 발리데이션 스플릿 0.2 랜덤= 388
# result : [0.6271927952766418, 0.7318313717842102]
# acc : 0.7318313640783801


#레이어 변경 * 배치 5k , 에포 100, 페이션스 10, 발리데이션 스플릿 0.2 랜덤= 388
# 변경했더니 완전 구림  인풋이 많은만큼 많은 노드가 필요한가??

#

#acc : 0.8736607488619055