import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_wine, load_digits, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

# 1. 데이터
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
y = train_csv['quality']-3

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state = 337, shuffle = True,
)
print(x_train.shape)
print(np.unique(y_train, return_counts=True))

# 2. 모델
def build_model(drop=0.2, optimizer='adam', activation='relu', node1=64, node2=64, node3=64, node4=64, lr = 0.001):
    inputs = Input(shape=(12,), name='input')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    outputs = Dense(7, name='outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=["accuracy"])    
    return model



def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    lr = [0.001, 0.005, 0.01]
    optimizers = [Adam(learning_rate=lr), RMSprop(learning_rate=lr), Adadelta(learning_rate=lr)]
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'lr' : lr}

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer, r2_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import time

start = time.time()
# model = GridSearchCV(estimator=KerasClassifier(build_fn=build_model), param_grid=create_hyperparameter(), cv=2)
model = RandomizedSearchCV(estimator=KerasClassifier(build_fn=build_model), param_distributions=create_hyperparameter(), cv=5, n_iter=1, verbose=1)
model.fit(x_train, y_train, epochs = 50)
end = time.time()

print('걸린시간 : ', end - start)
best_params = model.best_params_.copy()
best_params['optimizer'] = best_params['optimizer'].__class__.__name__
print('model.best_params_ :', best_params)
print('model.best_estimator : ', model.best_estimator_)
print('model.best_score_ : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc : ', accuracy_score(y_test, y_predict))
# print('r2 :', r2_score(y_test, y_predict))

# 걸린시간 :  21.74860692024231
# model.best_params_ :  {'optimizer': <keras.optimizer_v2.rmsprop.RMSprop object at 0x000001F7E2AF2940>, 'drop': 0.4, 'batch_size': 100, 'activation': 'elu'}
# model.best_estimator :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001FD83F48E80>
# model.best_score_ :  0.2731554489582777
# 11/11 [==============================] - 0s 900us/step - loss: 2.0950 - accuracy: 0.4136
# model.score :  0.41363635659217834
# acc :  0.41363636363636364