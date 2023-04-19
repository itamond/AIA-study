import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
#1. 데이터

x, y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

parameters = [
    {'rf__n_estimators':[100, 200]}, 
    {'rf__max_depth':[6, 8, 10, 12]},
    {'rf__min_samples_leaf':[3, 5, 7, 10]},
    {'rf__min_samples_split':[2, 3, 5, 10]},
]
parameters =[
    {'rf__n_estimators':[100],'rf__max_depth':[6,8,10,12],'rf__min_samples_leaf':[3,10],'rf__min_samples_split':[2,10]},
    {'rf__n_estimators':[100],'rf__max_depth':[6,8,10,12],'rf__min_samples_leaf':[5,7],'rf__min_samples_split':[3,5]},
    {'rf__n_estimators':[200],'rf__max_depth':[6,8],'rf__min_samples_leaf':[7,10],'rf__min_samples_split':[5,10]},
    {'rf__n_estimators':[200],'rf__max_depth':[10,12],'rf__min_samples_leaf':[3,5],'rf__min_samples_split':[2,3,]}    
]



#2. 모델
pipe = Pipeline([("std",StandardScaler()), ('rf',RandomForestClassifier())])
model = GridSearchCV(pipe, parameters,
                     cv = 5,
                     verbose=1,
                     n_jobs=-1
                     )

# Invalid parameter
# pipe의 parameter를 넣어주어야 에러가 뜨지 않는다.
# rf의 파라미터를 pipeline의 parameter 형태로 바꿔줘야한다.

stt = time.time()
#3. 훈련
model.fit(x_train, y_train)
ett = time.time()
#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
y_predict = model.predict(x_test)
print('ACC :', accuracy_score(y_test, y_predict))

print('최적의 매개변수 :', model.best_estimator_) #가장 좋은 평가

print('최적의 파라미터 :', model.best_params_) #가장 좋은 평가

print('best_score_ :', model.best_score_) #가장 좋은 평가

print('model.score :', model.score(x_test,y_test))

y_predict = model.predict(x_test)

y_pred_best = model.best_estimator_.predict(x_test) 

print('accuracy_score :', accuracy_score(y_test, y_predict))


print('최적 튠 ACC:', accuracy_score(y_test, y_pred_best))

print('걸린 시간 :', round(ett-stt,2),'초')

