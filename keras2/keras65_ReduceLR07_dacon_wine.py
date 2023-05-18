#ReduceLRonPlateau : 개선이 없으면 Learningrate 반으로 줄인다.


import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

#1. 데이터

path = './_data/dacon_wine/'
path_save = './_save/dacon/wine/'
train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)


# Remove rows with single class label
single_class_label = train_csv['quality'].nunique() == 1
if single_class_label:
    train_csv = train_csv[train_csv['quality'] != train_csv['quality'].unique()[0]]

# Label encode 'type'
le = LabelEncoder()
train_csv['type'] = le.fit_transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

# Split data
x = train_csv.drop(['quality'], axis=1)
y1 = train_csv['quality']-3

y = to_categorical(y1)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state = 337, shuffle = True,
)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=x.shape[1]))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(len(np.unique(y1)),activation='softmax'))

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate,)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor = 'val_loss',
                   patience = 20, mode = 'min',
                   verbose = 1)

rlr = ReduceLROnPlateau(monitor = 'val_loss',
                        patience = 2,
                        mode='auto',
                        verbose = 1,
                        factor=0.5)

model.fit(x_train, y_train,
          epochs = 200, 
          batch_size = 32, 
          validation_split=0.2, 
          verbose=1, 
          callbacks=[es, rlr])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_pred, y_test)

print("acc :", acc)



# acc : 0.5218181818181818
