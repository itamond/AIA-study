
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#1. 데이터

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

parameters = [
    {'randomforestclassifier__n_estimators':[100, 200]}, 
    {'randomforestclassifier__max_depth':[6, 8, 10, 12]},
    {'randomforestclassifier__min_samples_leaf':[3, 5, 7, 10]},
    {'randomforestclassifier__min_samples_split':[2, 3, 5, 10]},
]
parameters =[
    {'randomforestclassifier__n_estimators':[100],'randomforestclassifier__max_depth':[6,8,10,12],'randomforestclassifier__min_samples_leaf':[3,10],'randomforestclassifier__min_samples_split':[2,10]},
    {'randomforestclassifier__n_estimators':[100],'randomforestclassifier__max_depth':[6,8,10,12],'randomforestclassifier__min_samples_leaf':[5,7],'randomforestclassifier__min_samples_split':[3,5]},
    {'randomforestclassifier__n_estimators':[200],'randomforestclassifier__max_depth':[6,8],'randomforestclassifier__min_samples_leaf':[7,10],'randomforestclassifier__min_samples_split':[5,10]},
    {'randomforestclassifier__n_estimators':[200],'randomforestclassifier__max_depth':[10,12],'randomforestclassifier__min_samples_leaf':[3,5],'randomforestclassifier__min_samples_split':[2,3,]}    
]


#2. 모델
# pipe = Pipeline([("std",StandardScaler()), ('rf',RandomForestClassifier())])
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
model = GridSearchCV(pipe, parameters,
                     cv = 5,
                     verbose=1,
                     n_jobs=-1
                     )

# Invalid parameter
# pipe의 parameter를 넣어주어야 에러가 뜨지 않는다.
# rf의 파라미터를 pipeline의 parameter 형태로 바꿔줘야한다.


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
y_predict = model.predict(x_test)
print('ACC :', accuracy_score(y_test, y_predict))


