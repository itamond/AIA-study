#판다스 원핫을 이용해 사이킷런 와인 데이터 acc 모델링 하기
# 사이킷런 원핫을 이용해 acc 모델링 해보기
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
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

y = np.array(y)



# encoder =OneHotEncoder()
# y=encoder.fit_transform(y)



x_train, x_test, y_train, y_test =train_test_split(
    x, y,
    random_state=3342,
    shuffle=True,
    stratify=y,
    train_size=0.8,
)



scaler=MinMaxScaler()
# scaler=MaxAbsScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# print(np.unique(y_train, return_counts=True))

# print(y_test.shape)

# #2. 모델구성

# model=Sequential()
# model.add(Dense(20,input_dim=13))
# model.add(Dense(40,activation='relu'))
# model.add(Dense(60,activation='relu'))
# model.add(Dense(40,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(3,activation='softmax'))


input1 = Input(shape=(13,))
dense1 = Dense(20,activation='relu')(input1)
dense2 = Dense(40,activation='relu')(dense1)
dense3 = Dense(60,activation='relu')(dense2)
dense4 = Dense(40,activation='relu')(dense3)
dense5 = Dense(20,activation='relu')(dense4)
output1 = Dense(3,activation='softmax')(dense5)


model = Model(inputs=input1, outputs=output1)





#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics=['acc']
              )

es = EarlyStopping(monitor='val_acc',
                   mode='max',
                   patience=150,
                   verbose=1)


model.fit(x_train, y_train,
          epochs=3000,
          batch_size=5,
          verbose=1,
          validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print(results)
print('loss :', results[0])    # 리절트의 0번째
print('acc :', results[1])    #리절트의 1번째
y_pred = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)  # axis = 1 각 행에 있는 '열' 끼리 비교
y_pred = np.argmax(y_pred, axis=1)

# print(y_test)
acc = accuracy_score(y_test_acc, y_pred)
# y_test = np.drop(['id'])

# y_predict=model.predict(x_test)
# y_predict=np.argmax(y_predict,axis=1)




# loss : 0.08538362383842468
# acc : 0.9444444179534912    맥스앱스 스케일러 적용

# loss : 0.012840759009122849
# acc : 1.0    민맥스 스케일러 적용