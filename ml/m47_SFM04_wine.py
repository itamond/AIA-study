#가중치 역시 데이터이다.
#데이터(가중치)를 저장하는 방법에 대하여import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine,fetch_covtype,load_digits,fetch_california_housing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

x, y = load_wine(return_X_y=True)
parameter ={'n_estimators' : 1000,
            'learning_rate' : 0.3, #일반적으로 가장 성능에 영향을 많이 끼침. 경사하강법에서 얼만큼씩 하강할것이냐를 뜻함. 웨이트를 찾을때 적절한 러닝레이트 필요
            'max_depth' : 2, #트리형 모델의 깊이.
            'gamma' : 0,
            'min_child_weight' : 0, 
            'subsample' : 0.2, # 드랍아웃의 개념. 0.2만큼 덜어낸다는 의미
            'colsample_bytree' : 0.5,
            'colsample_bylevel': 0,
            'colsample_bynode': 0,
            'reg_alpha': 1, #알파와 람다 l1, l2 규제
            'reg_lambda': 1,
            'random_state': 337,
            }

#2. 모델
scaler = RobustScaler()
model = XGBClassifier(**parameter
                      )



x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=337,train_size=0.8)

model.fit(x_train, y_train, eval_set=[(x_train,y_train),(x_test,y_test)], 
          early_stopping_rounds=10,
          verbose=0,
          eval_metric='auc')



# print(model.feature_importances)
thresholds = np.sort(model.feature_importances_)

print(thresholds)
# [0.03175544 0.06501069 0.07197672 0.07753727 0.08700569 0.10525618
#  0.12046603 0.13875969 0.14712931 0.15510292]     10개의 컬런중 지정한 값과 나머지것들을 비교하여 지정한 값과 그 값보다 큰 값을 유지함. 따라서 for문에 입력하면 1번 값부터 값을 비교하며 지정한 값보다 낮은 값은 drop됨

for i in thresholds :
    selection = SelectFromModel(model,threshold=i,
                                prefit=True, #prefit 사전 훈련
                                # prefit=False, #사전훈련한 가중치를 사용하지 않는다는 의미. 다시 훈련시켜서 가중치를 사용함
                            )
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print('변형된 x_train :', select_x_train.shape, '변형된 x_test :', select_x_test.shape)
    
    selection_model = XGBClassifier()
    
    selection_model.set_params(**parameter, eval_metric='merror',)
    
    selection_model.fit(select_x_train,y_train,
                        # eval_set=[(select_x_train,y_train), (select_x_test, y_test)],
                        verbose=0, )
    
    select_y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)
    print("Tresh=%.3f, 남은 컬런 갯수=%d, R2: %.2f%%"%(i, select_x_train.shape[1], score*100))
    #%.3f는 float 소수 셋째짜리까지의 숫자를 넣으라는 의미, 그 숫자는 뒤 % 뒤로 정의한 내용중 첫번째.
    #%d는 정수형으로 값을 빼라는 의미, 그 숫자는 뒤 % 뒤로 정의한 내용중 두번째.
    #%.2f%% 는 float 소수 둘째자리까지의 숫자를 넣으라는 의미. %%는 글자 %를 입력하고 싶을때 쓰는 문법. 출력되는 숫자는 %뒤로 정의한 내용중 세번째

# 변형된 x_train : (142, 13) 변형된 x_test : (36, 13)
# Tresh=0.017, 남은 컬런 갯수=13, R2: 90.46%
# 변형된 x_train : (142, 12) 변형된 x_test : (36, 12)
# Tresh=0.018, 남은 컬런 갯수=12, R2: 95.23%
# 변형된 x_train : (142, 11) 변형된 x_test : (36, 11)
# Tresh=0.025, 남은 컬런 갯수=11, R2: 90.46%
# 변형된 x_train : (142, 10) 변형된 x_test : (36, 10)
# Tresh=0.026, 남은 컬런 갯수=10, R2: 95.23%
# 변형된 x_train : (142, 9) 변형된 x_test : (36, 9)
# Tresh=0.027, 남은 컬런 갯수=9, R2: 95.23%
# 변형된 x_train : (142, 8) 변형된 x_test : (36, 8)
# Tresh=0.031, 남은 컬런 갯수=8, R2: 90.46%
# 변형된 x_train : (142, 7) 변형된 x_test : (36, 7)
# Tresh=0.034, 남은 컬런 갯수=7, R2: 95.23%
# 변형된 x_train : (142, 6) 변형된 x_test : (36, 6)
# Tresh=0.047, 남은 컬런 갯수=6, R2: 90.46%
# 변형된 x_train : (142, 5) 변형된 x_test : (36, 5)
# Tresh=0.068, 남은 컬런 갯수=5, R2: 85.70%
# 변형된 x_train : (142, 4) 변형된 x_test : (36, 4)
# Tresh=0.088, 남은 컬런 갯수=4, R2: 85.70%
# 변형된 x_train : (142, 3) 변형된 x_test : (36, 3)
# Tresh=0.088, 남은 컬런 갯수=3, R2: 80.93%
# 변형된 x_train : (142, 2) 변형된 x_test : (36, 2)
# Tresh=0.093, 남은 컬런 갯수=2, R2: 57.09%
# 변형된 x_train : (142, 1) 변형된 x_test : (36, 1)
# Tresh=0.438, 남은 컬런 갯수=1, R2: 38.01%
