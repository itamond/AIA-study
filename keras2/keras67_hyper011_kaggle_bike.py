import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_wine, load_digits, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

# 1. 데이터


path='./_data/kaggle_bike/'
path_save='./_save/kaggle_bike/'    

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 

train_csv = train_csv.dropna()   

x = train_csv.drop(['count'], axis=1)   
y = train_csv['count']
x_train, x_test, y_train, y_test = tts(
    x,y, train_size=0.8, random_state = 337, shuffle = True,
)

print(x_train.shape)
# print(np.unique(y_train, return_counts=True))

# 2. 모델
def build_model(drop=0.2, optimizer='adam', activation='relu', node1=64, node2=64, node3=64, node4=64, lr=0.001):
    inputs = Input(shape=(10,), name='input')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    outputs = Dense(1, name='outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    
    model.compile(optimizer='adam', loss='mse')    
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
model = RandomizedSearchCV(estimator=KerasRegressor(build_fn=build_model), param_distributions=create_hyperparameter(), cv=5, n_iter=1, verbose=1)
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
# print('acc : ', accuracy_score(y_test, y_predict))
print('r2 :', r2_score(y_test, y_predict))

# 걸린시간 :  11.608684301376343
# model.best_params_ :  {'optimizer': <keras.optimizer_v2.adadelta.Adadelta object at 0x0000010CDCAFEDC0>, 'drop': 0.3, 'batch_size': 400, 'activation': 'linear'}
# model.best_estimator :  <keras.wrappers.scikit_learn.KerasRegressor object at 0x0000010CE48286A0>
# model.best_score_ :  -84.1928973197937
# 6/6 [==============================] - 0s 800us/step - loss: 8.0631
# model.score :  -8.063093185424805
# r2 : 0.9997630178679188