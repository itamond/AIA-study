#cnn 각각 함수형으로 만들기.
#House Prices - Advanced Regression Techniques
#라벨인코더 활용 - 스트링 변환
from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import MaxPooling1D, Dense, Conv1D, Flatten, Input, GRU, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import datetime
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath='./_save/cnn/mnist/'
filename='{epoch:04d}-{val_acc:.4f}.hdf5'



#2. 모델구성
input1 = Input(shape=(28, 28))
Conv1 = Conv1D(20, 2, padding='same')(input1)
Conv2 = Conv1D(10, 2)(Conv1)
Maxp = MaxPooling1D()(Conv2)
Conv3 = Conv1D(10, 2)(Maxp)
Flat1 = Flatten()(Conv3)
dense1 = Dense(10, activation='relu')(Flat1)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
output1 = Dense(10,activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)   




#3. 컴파일, 훈련
import time
start_time = time.time()


es = EarlyStopping(monitor='val_acc',
                   mode='auto',
                   restore_best_weights=True,
                   patience=20)

mcp = ModelCheckpoint(monitor='val_acc',
                      mode='auto',
                      save_best_only=True,
                      verbose=1,
                      filepath = ''.join([filepath+'_k32_2_'+date+'_'+filename]))


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

hist = model.fit(x_train,y_train,
                 epochs = 300,
                 batch_size = 64,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es])

end_time = time.time()

#4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result :', result)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_argm = np.argmax(y_test, axis=1)
acc = accuracy_score(y_argm, y_pred)

print('acc :', acc)

print('걸린시간 : ', round(end_time - start_time,2))    # round의 2는 소수 둘째까지 반환하라는것


# result : [0.19968438148498535, 0.9404000043869019]
# acc : 0.9404
# 걸린시간 :  136.27 Conv1D 적용

# result : [0.2009793519973755, 0.9426000118255615]
# acc : 0.9426
# 걸린시간 :  250.63   Conv1D적용\\
    
# result : [0.1158280000090599, 0.9667999744415283]
# acc : 0.9668
# 걸린시간 :  277.82   맥스풀링 1차 적용


#맥스풀링 후 Conv1D 한번 더 적용