from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.python.keras.models import Model, Sequential
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터

dataset = fetch_covtype()

# print(dataset.DESCR)

x = dataset.data
y = dataset.target

# print(x.shape, y.shape)  #(581012, 54) (581012,)

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    stratify=y,
                                                    random_state=333,
                                                    )

