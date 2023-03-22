from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(7, 
                 (2, 2),
                 padding='valid',
                 strides=1,#input 모양과 같은 모양으로 떨어트리는 패딩
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


#stride = 보폭. 즉 커널의 움직이는 칸 수를 의미한다.
#maxpooling의 디폴트 stride는 2이다. 그래서 겹치지 않는것