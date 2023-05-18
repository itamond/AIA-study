import autokeras as ak
import time
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist
import tensorflow as tf


#1. 데이터


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

# AutoKeras 분류 모델 생성
model = ak.StructuredDataRegressor(max_trials=2, overwrite=False)  # 최대 시도 횟수 지정

# 모델 훈련
model.fit(x_train, y_train, epochs=200, validation_split=0.15)

# 모델 평가
results = model.evaluate(x_test, y_test)
print('결과:', results)

# best_model = model.export_model()
# print(best_model.summary())