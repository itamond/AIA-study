#판다스 원핫을 이용해 사이킷런 와인 데이터 acc 모델링 하기
# 사이킷런 원핫을 이용해 acc 모델링 해보기
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder

#1. 데이터 

data_sets = load_wine()
x = data_sets.data
y = data_sets.target
# print(x.shape, y.shape)    (178, 13) (178,)

# print(data_sets.DESCR)
# print(y)

# print('y의 라벨값 :', np.unique(y))


#판다스 원핫 해보기

y = pd.get_dummies(y)
print(y.shape)


# encoder =OneHotEncoder()
# y=encoder.fit_transform(y)



x_train, x_test, y_train, y_test =train_test_split(
    x, y,
    random_state=3342,
    shuffle=True,
    stratify=y,
    train_size=0.8,
)

# print(np.unique(y_train, return_counts=True))

# print(y_test.shape)

#2. 모델구성

model=Sequential()
model.add(Dense(20,input_dim=13))
model.add(Dense(40,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(3,activation='softmax'))




#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics=['acc']
              )

es = EarlyStopping(monitor='val_acc',
                   mode='max',
                   patience=25,
                   verbose=1)


model.fit(x_train, y_train,
          epochs=3,
          batch_size=5,
          verbose=1,
          validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :',result)


y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict,axis=1)


print(y_predict)
