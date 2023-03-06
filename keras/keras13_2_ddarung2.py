# 데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error  #rmse 함수를 직접 만드는 법 
import pandas as pd # numpy 만큼 많이 나오는 자료형, 전처리 하는 내용
#csv 파일을 판다스를 이용하면 깔끔하게 땡겨올 수 있음. 실무에서 계속 사용


#1. 데이터
path='./_data/ddarung/'      # .=현 폴더, study    /= 하위폴더

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 #index_col = x번째 컬런이 index 라고 지정하는 함수
                                                     #read_csv 파일을 읽어오는 함수, 문자로 되어 있으니 따옴표
#train_csv = pd.read_csv('./_data/ddarung/train.csv')

# print(train_csv.shape) # (1459, 10)
# # id 는 데이터가 아니다. id를 삭제 해줘야함

# print(train_csv)  #판다스에는 이름과 컬런명이 따라다님. 하지만 컬런 '이름'은 데이터로 들어가지 않는다.
#                   #index도 연산하지 않는다. 때문에 index 컬런을 지정해줘야함

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

# print(test_csv)
# print(test_csv.shape)

#=================================================================

print(train_csv.columns)


# #Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())    #데이터 종류를 보는 판다스 함수
#print(train_csv.describe())  
#중단점 찍어보기   (번호 왼쪽에 빨간 점을 찍으면 거기까지만 실행한다.)
#그냥 F5만 눌러서 디버그 실행 해보자.

#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64





print(type(train_csv))    #<class 'pandas.core.frame.DataFrame'> 타입도 항상 확인해라


######################################결측치 처리###############################################
#통 데이터일 때 결측치 처리를 한다. 분리 후 결측치 처리를 하게되면 데이터가 망가진다.

# 결측치 처리 1. 제거                (결측된 데이터의 행 자체가 사라지기 때문에 아까운 방법)
#print(train_csv.isnull())   # isnull   -> 데이터가 null값인가요? 하고 물어보는 함수
print(train_csv.isnull().sum())  #isnull의 트루값이 몇개인지에 대한 합계(sum) ************자주 사용한다.
train_csv = train_csv.dropna()   #dropna = 결측치 삭제 함수*****
print(train_csv.isnull().sum())
print(train_csv.info())          #데이터 몇개가 남았는지 확인 
print(train_csv.shape)           #(1328, 10)




############################train_csv 데이터에서 x와 y를 분리(매우 중요)#########################
x = train_csv.drop(['count'], axis=1)     #drop 버리기 함수 , 따옴표 안에 열 이름 axis = 0이 행 1이 열
print(x)

y = train_csv['count']
print(y)
###############################################################################################
#판다스는 자료형중에서 index와 column을 가지고 있다.



x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    train_size=0.7,
    random_state=221
)
                                                                   #결측치 제거 후
print(x_train.shape, x_test.shape)  #(1021, 9) (438, 9)           ->(929, 9) (399, 9)
print(y_train.shape, y_test.shape)  #(1021,) (438,)               ->(929,) (399,)





#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train,
          epochs= 100,
          batch_size=16,
          verbose=1)

#4. 평가, 예측

loss= model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)


def RMSE(y_test, y_predict):             #함수의 약자 def(함수를 만드는 함수) 임의로 RMSE라고 지었다   함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))    #return = 반환시키는 함수,   np.sqrt = 넘파이 스쿼트. 루트 씌우는 함수
#이 자체로 실행되지는 않음.

rmse = RMSE(y_test, y_predict)     #위에서 정의한 RMSE 함수 사용

print('RMSE :', rmse)


##### submission.csv를 만들어봅시다!!! #####
# 위 모델에서 만들어진 w값을 토대로 test.csv를 대입하며 count 를 구한다.
# print(test_csv.isnull().sum())  #여기도 결측치가 있다.
y_submit=model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path+'submission.csv', index_col=0)
print(submission)
submission['count'] = y_submit     #submission의 count 열에 y_submit을 입력
print(submission)
# 이것을 다시 파일로 저장

submission.to_csv(path + 'submit_0306_0447.csv')     #저장하는 함수 .to_
