
import numpy as np
import pandas as pd
import datetime
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
#1. 데이터

path = './_data/basic/'
path_save = './_save/basic/'

train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv',index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/keras28/11/'
filename = '{epoch:04d}-{val_acc:.2f}.hdf5'

# print(train_set.describe) 
# print(train_set.columns)



x=train_set.drop(['전화해지여부'], axis=1)
y=train_set['전화해지여부']
print(x.shape, y.shape)  #(30200, 12) (30200,)




# print(np.unique(y, return_counts=True))

# x = datasets['data']     #딕셔너리의 key
# y = datasets.target
# # print(x.shape, y.shape)  #(569, 30) (569,)

# # print(datasets.isnull().sum())
# # print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,
    random_state=37637,
    test_size=0.1,
    # stratify=y,
)
# scaler=RobustScaler()
# scaler=MaxAbsScaler()

scaler=RobustScaler()
# scaler=MaxAbsScaler()
# scaler=StandardScaler()
# scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)

train_x=x_train

train_x['총통화시간'] = train_x['주간통화시간'] + train_x['저녁통화시간'] + train_x['밤통화시간']

train_x['총통화횟수'] = train_x['주간통화횟수'] + train_x['저녁통화횟수'] + train_x['밤통화횟수']

train_x['총통화요금'] = train_x['주간통화요금'] + train_x['저녁통화요금'] + train_x['밤통화요금']

train_x['평균통화시간'] = train_x['총통화시간'] / train_x['총통화횟수']
train_x['평균통화요금'] = train_x['총통화요금'] / train_x['총통화횟수']




#2. 모델구성

# model = Sequential()
# model.add(Dense(8, activation='linear',input_dim=8))
# model.add(Dense(6,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(6,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))  


input1 = Input(shape=(10,))
dense1 = Dense(200,activation='linear')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(100,activation='relu')(drop1)
drop2 = Dropout(0.5)(dense2)
dense3 = Dense(40,activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(10,activation='relu')(drop3)
output1 = Dense(1,activation='sigmoid')(dense4)

model = Model(inputs=input1, outputs=output1)
class_weight={0:10, 1:80}

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
              # 훈련 과정에 accuracy를 추가.   hist.history에 다 기록됨. 대괄호=리스트
              )                             #대괄호 안에 다른 지표도 추가 가능(list이기 때문에)
                                            #metrics = 훈련에 영향을 주지않음. 지표확인용, 풀네임 줄임말 상관 없음

mcp = ModelCheckpoint(monitor='val_acc',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=''.join([filepath,'basic_1_',date,'_',filename]),
                      )



es =EarlyStopping(monitor='val_acc',
                  mode='auto',
                  patience=100,
                  restore_best_weights=True,
                  verbose=1,
                  )



hist = model.fit(x_train,y_train,
                 epochs=5000,
                 batch_size=64,
                 validation_split=0.3,
                 verbose=1,
                 callbacks=[es,mcp],
                 class_weight=class_weight
                 )


#4. 평가, 예측
result = model.evaluate(x_test,y_test)   
print('result :', result)

y_predict= np.round(model.predict(x_test))


# r2= r2_score(y_test, y_predict)
# print('r2 :', r2)

def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, y_predict)
print('rmse :', rmse)

#평소처럼 수치가 아닌 0이냐 1이냐를 맞춰야한다면 accuracy_score 사용 



from sklearn.metrics import accuracy_score, r2_score
acc = accuracy_score(y_test, y_predict)
print('acc :', acc)

macro_f1 = f1_score(y_test, y_predict, average='macro')
print("Macro F1 Score: {:.2f}".format(macro_f1))
y_submit = np.round(model.predict(test_set))
submission['전화해지여부']=y_submit
submission.to_csv(path_save+'basic_submission_01.csv')

macro_f1 = f1_score(y_test, y_predict, average='macro')

print("Macro F1 Score: {:.2f}".format(macro_f1))

