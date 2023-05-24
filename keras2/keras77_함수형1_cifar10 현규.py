import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def m1():
    input1 = Input(1)
    dense1 = Dense(3)(input1)
    dense2 = Dense(2)(dense1)
    output1 = Dense(1)(dense2)
    model = Model(inputs=input1, outputs=output1)
    
    model.layers[0].trainable = False # hidden1
    pd.set_option('max_colwidth', -1)

    layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
    # print(layers)

    results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
    print(results) # 가중치 False

def m2():
    input1 = Input(1)
    dense1 = Dense(3)(input1)
    dense2 = Dense(2)(dense1)
    output1 = Dense(1)(dense2)
    model = Model(inputs=input1, outputs=output1)
    
    model.layers[1].trainable = False # hidden1
    pd.set_option('max_colwidth', -1)

    layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
    # print(layers)

    results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
    print(results) # 가중치 False

def m3():
    input1 = Input(1)
    dense1 = Dense(3)(input1)
    dense2 = Dense(2)(dense1)
    output1 = Dense(1)(dense2)
    model = Model(inputs=input1, outputs=output1)
    
    model.layers[2].trainable = False # hidden1
    pd.set_option('max_colwidth', -1)

    layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
    # print(layers)

    results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
    print(results) # 가중치 False

def_list = [m1,
            m2,
            m3]
for d in range(len(def_list)):
    if d == 0:
        m1()
    elif d == 1:
        m2()
    elif d == 2:
        m3()