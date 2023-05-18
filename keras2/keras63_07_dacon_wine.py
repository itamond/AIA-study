import time
import autokeras as ak
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder
import pandas as pd

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


model = ak.StructuredDataClassifier(max_trials=2, overwrite=False)  # 최대 시도 횟수 지정

# 모델 훈련
model.fit(x_train, y_train, epochs=200, validation_split=0.15)

# 모델 평가
results = model.evaluate(x_test, y_test)
print('결과:', results)
