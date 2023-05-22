import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255.


# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=64, node2=64, node3=64, node4=64, lr = 0.001):
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(node1, 2, activation=activation, name='Conv2D')(inputs)
    x = Flatten()(x)
    x = Dense(node1, activation=activation, name='hidden1')(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    
    model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')    
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
from sklearn.metrics import accuracy_score, make_scorer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import time

es = EarlyStopping(monitor='val_accuracy',
                   patience=5,
                   mode='auto',
                   restore_best_weights=True)

# mcp = ModelCheckpoint(monitor='loss',
#                       patience=5,
#                       )

start = time.time()
# model = GridSearchCV(estimator=KerasClassifier(build_fn=build_model), param_grid=create_hyperparameter(), cv=2)
model = RandomizedSearchCV(estimator=KerasClassifier(build_fn=build_model), param_distributions=create_hyperparameter(), cv=5, n_iter=1, verbose=1)
model.fit(x_train, y_train, epochs = 50, callbacks=[es], validation_split = 0.2)
end = time.time()

print('걸린시간 : ', end - start)
print('model.best_params_ : ', model.best_params_)
print('model.best_estimator : ', model.best_estimator_)
print('model.best_score_ : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc : ', accuracy_score(y_test, y_predict))

# model.best_params_ :  {'optimizer': <keras.optimizer_v2.adam.Adam object at 0x000001E41E748C40>, 'drop': 0.4, 'batch_size': 500, 'activation': 'elu'}
# model.best_estimator :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E492C89B50>
# model.best_score_ :  0.9664166569709778
# model.score :  0.9718000292778015
# acc :  0.9718