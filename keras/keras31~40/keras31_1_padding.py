from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(7, 
                 (2, 2),
                 padding='same',              #input 모양과 같은 모양으로 떨어트리는 패딩
                 input_shape=(8,8,1)))   
model.add(Conv2D(filters=4,
                 kernel_size=(4,4),
                 padding='same',             #패딩의 디폴트 = valid. 없는것과 같음. same 패딩의 패딩 사이즈는 커널 사이즈에 비례하여 커진다.
                 activation='relu'))    
model.add(Conv2D(10, (2,2)))                
model.add(Flatten())                    
model.add(Dense(32, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))


model.summary()


#레이어를 거쳐갈 때마다
#커널 사이즈 -1 만큼 줄어듬
#필터를 적게 거쳐간 외곽은 데이터가 소실됨.
#필터를 많이 거쳐갈수록 유실되는 데이터가 많고, 특징이 사라진다.
#이러한 과정을 거칠때 소실되는 외곽의 데이터를 최소화 하기 위해 padding을 씌운다.
#이 패딩은 0의 데이터를 채워넣어 연산에 영향을 주지 않게 한다.
#0의 데이터가 행과 열에 추가되므로 레이어를 거쳐도 행과 열이 수축하지 않는다.
#padding 또한 하이퍼 파라미터 튜닝에 들어간다.(개발자가 판단)
