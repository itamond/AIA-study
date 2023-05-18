import autokeras as ak
import time
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist
import tensorflow as tf


#1. 데이터


data = load_iris()
x = data.data
y = data.target

# 데이터 분할
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=42)

# AutoKeras 분류 모델 생성
model = ak.StructuredDataClassifier(max_trials=2, overwrite=False)  # 최대 시도 횟수 지정

# 모델 훈련
model.fit(x_train, y_train, epochs=200, validation_split=0.15)

# 모델 평가
results = model.evaluate(x_test, y_test)
print('결과:', results)

# best_model = model.export_model()
# print(best_model.summary())