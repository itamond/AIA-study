# save_model과 비교


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler   


#1. 데이터

datasets = load_boston()
x= datasets.data
y= datasets['target']


x_train, x_test, y_train, y_test = train_test_split (x,y,
                                                     train_size=0.8,
                                                     random_state=333,
                                                     )


scaler = StandardScaler()

x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)            
print(np.min(x_test), np.max(x_test))      





#2. 모델



input1 = Input(shape=(13,))        
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)




#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam',)

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   mode='min',
                   verbose=1,
                #    restore_best_weights=True,
                   )

mcp = ModelCheckpoint(monitor='val_loss',
        mode='auto',
        verbose=1,
        save_best_only=True,
        filepath='./_save/MCP/keras27_3_MCP.hdf5'  #확장자는 국룰 느낌
)

model.fit(x_train,y_train,
        epochs=10000,
        batch_size=32,
        verbose=1,
        validation_split=0.2,
        callbacks=[es, mcp],
        )




model.save('./_save/MCP/keras27_3_save_model.h5')   



#4. 평가, 예측
from sklearn.metrics import r2_score
print("==================== 1. 기본 출력 ======================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


print("==================== 2. load_model 출력 ======================")
model2 =load_model('./_save/MCP/keras27_3_save_model.h5')
loss = model2.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


print("==================== 3. MCP 출력 ======================")
model3 =load_model('./_save/MCP/keras27_3_MCP.hdf5')
loss = model3.evaluate(x_test, y_test, verbose=0)
print('loss :', loss)

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# ==================== 1. 기본 출력 ======================
# 4/4 [==============================] - 0s 711us/step - loss: 22.7851
# loss : 22.785091400146484
# r2스코어 : 0.7676862822832251
# ==================== 2. load_model 출력 ======================
# 4/4 [==============================] - 0s 984us/step - loss: 22.7851
# loss : 22.785091400146484
# r2스코어 : 0.7676862822832251
# ==================== 3. MCP 출력 ======================
# 4/4 [==============================] - 0s 729us/step - loss: 22.7851
# loss : 22.785091400146484
# r2스코어 : 0.7676862822832251


#restore_best_weights 제거
#12번이 값이 더 좋은이유는 test 데이터 문제이지 트레인 때의 w값은 3번이 가장 좋다
# ==================== 1. 기본 출력 ======================
# loss : 25.152570724487305
# r2스코어 : 0.7435477750235517
# ==================== 2. load_model 출력 ======================
# loss : 25.152570724487305
# r2스코어 : 0.7435477750235517
# ==================== 3. MCP 출력 ======================
# loss : 24.21658706665039
# r2스코어 : 0.7530909334104068