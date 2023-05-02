#라벨링 한 데이터를 스케일링 하는가?
#라벨인코딩 했을경우 스케일링 할 필요 없다.
#카테고리 형태이니 라벨인코딩 해준다.
#원핫 안하고도 돌릴수는 있다.
#원핫하게되면 컬런 갯수가 많이 늘어남.
#PCA,LDA 사용할 수 있음...
#시간 데이터도 23시 이후에 0시가 된다는건 서로의 상하관계가 아닌 연속성으로 봐야한다.
#따라서 sin을 활용 해야하는데..?


import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import time

#   / // \ \\ 같다
# \n = 줄바꿈과 같이 예약어가 되어버리면 쓸수 없음. 
# \\ 두개 넣으면 문제 없다.

path = './_data/finedust/'
'''
TRAIN 
TRAIN_AWS
TEST_INPUT
TEST_AWS
META 
answer_sample.csv
'''


# glob = 폴더 내의 모든 데이터를 가져와서 텍스트화 시켜줌

train_files = glob.glob(path + "TRAIN/*.csv")  #애스터리스크 = 모든것
# print(train_files)
test_input_files = glob.glob(path + 'test_input/*.csv') #경로에서는 대소문자 상관없다.
# print(test_input_files)
# 경로와 파일명이 리스트 형태로 구성되어 있음.
# 리스트 형태니까 for문을 통해 불러오기 가능


#########################Train폴더###############################
li = []

for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig') #인덱스는 없다. 제일 위에 0번째 행에 헤더가 있어서 0으로 지정. 한글 인코딩 지정
    li.append(df)
    
# print(li)
#리스트 안에 17개(지역)의 DATAFRAME([35064 rows x 4 columns]]이 모여있는 구조
# print(len(li))

train_dataset = pd.concat(li, axis = 0,
                          ignore_index = True)     # 행단위로 컨켓, 인덱스가 새로 부여되었기 때문에, ignore인덱스 함수 사용하여 인뎃스 삭제
# print(train_dataset) # [596088 rows x 4 columns]



#########################Test폴더###############################

li=[]
for filename in test_input_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig') #인덱스는 없다. 제일 위에 0번째 행에 헤더가 있어서 0으로 지정. 한글 인코딩 지정
    li.append(df)
    
# print(li)
#리스트 안에 17개(지역)의 DATAFRAME([35064 rows x 4 columns]]이 모여있는 구조
# print(len(li)) #[7728 rows x 4 columns]

test_input_dataset = pd.concat(li, axis = 0,
                          ignore_index = True)     # 행단위로 컨켓, 인덱스가 새로 부여되었기 때문에, ignore인덱스 함수 사용하여 인뎃스 삭제
# print(test_input_dataset) # [131376 rows x 4 columns]


####################측정소 라벨인코더##########################

le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소']) #트레인셋과 라벨을 동일시 하기위해 transform만 사용
# print(train_dataset) # [596088 rows x 5 columns]
# print(test_input_dataset) #[131376 rows x 5 columns]


train_dataset = train_dataset.drop(['측정소'],axis=1)
test_input_dataset = test_input_dataset.drop(['측정소'],axis=1)

# print(train_dataset)   #[596088 rows x 4 columns]
# print(test_input_dataset)  #[131376 rows x 4 columns]





##################일시 - > 월, 일, 시간으로 분리#################
# 12-31 21:00 - > 12 와 21 추출
print(train_dataset.info())

#  #   Column  Non-Null Count   Dtype
# ---  ------  --------------   -----
#  0   연도      596088 non-null  int64
#  1   일시      596088 non-null  object  -> 오브젝트 타입은 일반적으로 스트링이라고 생각.
#  2   PM2.5   580546 non-null  float64

#  3   locate  596088 non-null  int32
train_dataset['month'] = train_dataset['일시'].str[:2] #'일시'라는 string에 2번째, 즉 연도 숫자 두개만  뽑는다
train_dataset['hour'] = train_dataset['일시'].str[6:8] #'일시'라는 string에 2번째, 즉 연도 숫자 두개만  뽑는다
test_input_dataset['month'] = test_input_dataset['일시'].str[:2] #'일시'라는 string에 2번째, 즉 연도 숫자 두개만  뽑는다
test_input_dataset['hour'] = test_input_dataset['일시'].str[6:8] #'일시'라는 string에 2번째, 즉 연도 숫자 두개만  뽑는다
# print(train_dataset['hour'])
# print(train_dataset) #[596088 rows x 6 columns] 생성한 두개의 컬런이 추가됨.

train_dataset = train_dataset.drop(['일시'],axis=1)

### str - > int  month와 hour가 str 형태로 되어있으니 int로 변환

# train_dataset['month'] = pd.to_numeric(train_dataset['month'])
# train_dataset['month'] = train_dataset['month'].astype('int32')
#month 데이터는 12개의 숫자만 있기 때문에 int8로 하면 데이터 줄일수 있음. 하지만 데이터가 손상됨.
train_dataset['month'] = pd.to_numeric(train_dataset['month']).astype('int16')
train_dataset['hour'] = pd.to_numeric(train_dataset['hour']).astype('int16')
test_input_dataset['month'] = pd.to_numeric(test_input_dataset['month']).astype('int16')
test_input_dataset['hour'] = pd.to_numeric(test_input_dataset['hour']).astype('int16')

test_input_dataset = test_input_dataset.drop(['일시'],axis=1)


# print(train_dataset.info())
# print(test_input_dataset.info())


# print(train_dataset.info())
#  #   Column  Non-Null Count   Dtype
# ---  ------  --------------   -----
#  0   연도      596088 non-null  int64
#  1   PM2.5   580546 non-null  float64  결측치 있음. 그러나 다른 트레인 데이터가 월등히 많기때문에 깔끔히 드롭시키는것도 방법이다.
#  2   locate  596088 non-null  int32
#  3   month   596088 non-null  int16
#  4   hour    596088 non-null  int16


########################결측치 제거########################### PM2.5에 15542개
#전체 596085 -> 580546 으로 줄인다.
train_dataset = train_dataset.dropna()
print(train_dataset.info())


#파생 feature 만드는것을 항상 고민해야함 ex ) 공휴일, 계절 등등 *************상당히 중요함


# Data columns (total 5 columns):
#  #   Column  Non-Null Count   Dtype
# ---  ------  --------------   -----
#  0   연도      580546 non-null  int64
#  1   PM2.5   580546 non-null  float64
#  2   locate  580546 non-null  int32
#  3   month   580546 non-null  int16
#  4   hour    580546 non-null  int16

y = train_dataset['PM2.5']
x = train_dataset.drop(['PM2.5'], axis = 1)  #메모리에 문제 생길수도 있으니... 드롭을 나중에 한다


print(train_dataset)
# print(x, '\n', y) #\t는 탭 \n 줄바꿈

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state = 425, shuffle =True,
)

parameter = {
    'n_estimators' : 30000,
    'learning_rate' : 0.001, #일반적으로 가장 성능에 영향을 많이 끼침. 경사하강법에서 얼만큼씩 하강할것이냐를 뜻함. 웨이트를 찾을때 적절한 러닝레이트 필요
    'max_depth' : 12, #트리형 모델의 깊이.
    'gamma' : 0,
    'min_child_weight' : 0, 
    'subsample' : 0.2, # 드랍아웃의 개념. 0.2만큼 덜어낸다는 의미
    'colsample_bytree' : 0.5,
    'colsample_bylevel': 0,
    'colsample_bynode': 1,
    'reg_alpha': 1, #알파와 람다 l1, l2 규제
    'reg_lambda': 1,
    'random_state': 33610,
    'verbose' : 0,
    'n_jobs' : -1
}

# 2. 모델
model = XGBRegressor()

model.set_params(**parameter,
                 eval_metric='mae',      #컴파일하는 파라미터를 set_params에 넣는 구조
                 early_stopping_rounds=200,
                 )

stt = time.time()

model.fit(x_train, y_train, verbose =1,
          eval_set=[(x_train, y_train), (x_test, y_test)]    
)

ett = time.time()


#트루 테스트 생성
true_test = test_input_dataset[test_input_dataset['PM2.5'].isnull()].drop('PM2.5',axis=1)

# print(true_test)

#4. 평가, 예측
# y_predict = model.predict(x_test)
y_predict = model.predict(true_test)

results = model.score(x_test, y_test)
print('model.score :', results)

# r2 = r2_score(y_test, y_predict)
# print('r2 :', r2)

# mae = mean_absolute_error(y_test, y_predict)
# print('mae :', mae)

print('걸린시간 :', np.round((ett-stt),2),'초')

submission = pd.read_csv(path+'answer_sample.csv', index_col=0)
submission['PM2.5'] = y_predict
submission.to_csv(path+'sample1.csv')