from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#1. 데이터

datasets = load_boston()
x = datasets['data']
y = datasets['target']


print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    random_state=123,
    test_size=0.2,
    )


#2. 모델 구성
model = Sequential()
model.add(Dense(10,activation='relu', input_dim=13))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs = 200, verbose=1, batch_size=4,
                 validation_split=0.2)   # validation_split 은 train에서 땡겨온다

print(hist.history)

plt.plot(hist.history['loss'])   #x가 순서대로 갈 경우 명시하지 않아도 된다.
plt.show()


#  =epochs 순서대로 하나씩 저장된다


#훈련이 잘 되고있는지 


#4. 평가. 예측