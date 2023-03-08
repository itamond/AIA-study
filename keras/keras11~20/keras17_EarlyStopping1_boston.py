from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_squared_error

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
model.add(Dense(30,activation='relu', input_dim=13))     #sigmoid = 0~10까지 숫자로 한정시키는 액티베이션 함수
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
#땡겨온 얼리스토핑에 대한 정의
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',   # monitor = 나는 val_loss를 기준으로 할 것이다.(일반적으로 val_loss가 loss보다 나음) , patience=참을성. 5번 참겠다는 의미
              verbose=1,                                            # verbose=1을 쳐주면 어디서 stop 했는지 출력->Epoch 00011: early stopping
              restore_best_weights=True,                                # 최소의 loss 지점, 얼리 스톱된 지점의 w값을 저장하는 명령어
              )
                                                            # mode='min' 최소값 찾는 모드, r2값을 기준으로 한다면 최대값(max) 찾기(tensorflow에선 r2 지원안함), 
                                                            # mode='auto' 지표에 따라 자동으로 설정됨.
                                                            # 

hist = model.fit(x_train, y_train, epochs = 2000, verbose=1,
                 batch_size=16,
                 validation_split=0.2,
                 callbacks=[es])   #callbacks= 호출하다. es를 호출해라 # validation_split 은 train에서 땡겨온다



#모델.핏에서 반환하는 내용
print("================================================================")
print(hist)
print("================================================================")
print(hist.history) #로스, 발로스
print("================================================================")
print(hist.history['loss'])
print("================================================================")
print(hist.history['val_loss'])
print("================================================================")

# 4. 평가. 예측

loss=model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
r2 =  r2_score(y_test, y_predict)
print('r2스코어 :', r2)





#plt.rc('font', family=Malgun Gothinc')           #폰트 변환하는 두가지 방법
plt.rcParams['font.family'] = 'Malgun Gothic'     #맑은 고딕으로 폰트 변환   or  matplotlib.rcParams['font.family'] = 'Malgun Gothinc'
#가급적이면 폰트는 나눔체 써라. 저작권 문제

plt.figure(figsize=(9, 6))     #figsize = 피규어의 사이즈
plt.plot(hist.history['loss'], marker='.', c= 'red', label='로스')   #x가 순서대로 갈 경우 명시하지 않아도 된다.    
#marker= 선의 형태(.......),   c= 색상    , label = 라벨
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('보스턴')     # 그래프의 타이틀 
plt.xlabel('epochs')    #x축의 라벨을 지정한 스트링으로 표시
plt.ylabel('loss, val_loss') # y축의 라벨을 지정한 스트링으로 표시
plt.legend() # 선에대한 라벨명을 표시하는 함수
plt.grid() #격자 넣기 
plt.show()

#  =epochs 순서대로 하나씩 저장된다


# 훈련이 잘 되고있는지 


