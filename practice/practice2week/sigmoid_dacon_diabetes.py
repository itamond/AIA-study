#데이콘 당뇨 데이터를 이용하여 애큐러시 구하기

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_sets = pd.read_csv(path + 'train.csv', index_col=0)
test_sets=pd.read_csv(path + 'test.csv', index_col=0)
submission=pd.read_csv(path + 'sample_submission.csv', index_col=0)

# print(train_sets.isnull().sum())

x = train_sets.drop(['Outcome'],axis=1)
y = train_sets['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    stratify=y,
    random_state=339,
)


# print(x.shape, y.shape)

#2. 모델구성

model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

es = EarlyStopping(monitor='val_acc',
                   patience=20,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True)

hist = model.fit(x_train, y_train,
                 epochs=350,
                 batch_size=10,
                 callbacks=[es],
                 validation_split=0.2
                 )

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result :', result)

y_predict = np.round(model.predict(x_test))

def RMSE(a,b) :
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, y_predict)

print('rmse :', rmse)

acc = accuracy_score(y_test, y_predict)
print('acc :', acc)