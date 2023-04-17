# # 데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error  
import pandas as pd 



#1. 데이터
path='./_data/kaggle_bike/'
path_save='./_save/kaggle_bike/'    

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 
                                                     

# print(train_csv.shape)


# print(train_csv)  

# test_csv = pd.read_csv(path + 'test.csv',
#                        index_col=0)

# print(test_csv)
# print(test_csv.shape)

#=================================================================

# print(train_csv.columns)



# print(train_csv.info())    





# print(type(train_csv))   


######################################결측치 처리###############################################

#print(train_csv.isnull())   # isnull   -> 데이터가 null값인가요? 하고 물어보는 함수
print(train_csv.isnull().sum())  
train_csv = train_csv.dropna()   
print(train_csv.isnull().sum())
print(train_csv.info())          
print(train_csv.shape)          




############################train_csv 데이터에서 x와 y를 분리(매우 중요)#########################
x = train_csv.drop(['count'], axis=1)   
print(x)

y = train_csv['count']
print(y)
###############################################################################################




x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    train_size=0.7,
    random_state=221
)

print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 





#2. 모델구성
# model = Sequential()
# model.add(Dense(18, input_dim=10))
# model.add(Dense(27))
# model.add(Dense(36))
# model.add(Dense(45))
# model.add(Dense(36))
# model.add(Dense(27))
# model.add(Dense(18))
# model.add(Dense(9))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train,
#           epochs= 10,
#           batch_size=5,
#           verbose=100)

# #4. 평가, 예측

# loss= model.evaluate(x_test, y_test)
# print('loss :', loss)

# y_predict= model.predict(x_test)


from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor


models = [RandomForestRegressor(),DecisionTreeRegressor()]

for j, model in enumerate(models):
    model.fit(x,y)
    score = model.score(x,y)
    print(f" model {j+1}: {score:.3f}")