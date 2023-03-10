#y값이 실수가 아닌 0과 1일때의 모델링

#이진분류인 데이터

#나중에는 컬런에 대한 분석을 해야한다

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#1. 데이터

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'
train_sets = pd.read_csv(path + 'train.csv', index_col=0)
test_sets = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# print(train_sets.isnull().sum())

x = train_sets.drop(['Outcome'],axis=1)
y = train_sets['Outcome']

print(x.shape, y.shape)   #(652, 8) (652,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    stratify=y,
                                                    random_state=331,
                                                    )

#2. 모델 구성

model=Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['acc'])

es = EarlyStopping(monitor='acc',
                   patience=50,
                   restore_best_weights=True,
                   verbose=1,
                   validation_split=0.2,
                   mode='auto')

hist = model.fit(x_train,y_train,
                 epochs= 1000,
                 batch_size=10,
                 verbose=1,
                 callbacks=[es])

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc= accuracy_score(y_test, y_predict)
print('acc :', acc)



