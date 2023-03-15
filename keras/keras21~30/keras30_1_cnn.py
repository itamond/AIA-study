from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

# model=Sequential()
# model.add(Conv2D(7, 
#                  (2, 2), 
#                  input_shape=(5,5,1))) # 출력 : (N, 4, 4, 7)
#                                        # batch_size, rows, columns, channels  인풋 쉐이프의 정식 명칭

# #input_shape 이미지의 형태, 가로 5 세로 5, 1장(흑백)  , 3장이면 컬러(각각 RGB)
# #(2,2) = 자르는 크기(필터). 앞의 7 = > 필터를 거쳐 특성이 좁혀진 (4,4,1)이 7장이되어 (4,4,7)이 됨

# model.add(Conv2D(filters=4, 
#                  kernel_size=(3,3),
#                  activation='relu'))    # 출력 : (N, 2, 2, 4)       4,4,7의 데이터가 (3,3)의 필터를 거쳐 (2,2)가 되고, 최종 아웃풋 4개로 수렴한다.
# #컨볼루션은 4차원 데이터를 받아 4차원 데이터를 배출함
# #CNN의 노드 = filters

model=Sequential()
model.add(Conv2D(7, 
                 (2, 2), 
                 input_shape=(8,8,1)))   #출력 : (N, 7, 7, 7)
model.add(Conv2D(filters=4,
                 kernel_size=(3,3),
                 activation='relu'))     #출력 : (N, 5, 5, 4)
model.add(Conv2D(10, (2,2)))              #출력 : (N, 4, 4, 10)       #결국은 분류 모델에는 softmax로 끝나야 해서 컬런 갯수만큼의 output으로 바꿔야함
#(2,2)는 2x2, 즉 네개의 데이터가 있음. 그게 10개.즉 40개의 데이터가 있음. 이걸 softmax에 연결하기위해 DNN으로 바꾸려면, 40개의 노드로 펴줘야한다.
#(N, 4, 4 ,10) 은, 4x4가 10장. 따라서 160개
model.add(Flatten())                     #데이터를 펴주는 함수    #출력 : (N, 4*4*10) = (N, 160)   연산하지 않고 단순히 모양만 바꾼다.
model.add(Dense(32, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))


model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 7, 7, 7)           35            param = (2x2)(필터 크기) x 1(입력 채널 rgb) x 7 (출력 채널) + bias (출력 채널의 갯수)   =4x1x7 + 7   =35
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 5, 5, 4)           256           param = 3x3(필터 크기) x 7(입력 필터) x 4(출력 채널) + bias (출력 채널의 갯수) = 9 x 7 x 4 + 4    = 256
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 4, 10)          170           
# _________________________________________________________________
# flatten (Flatten)            (None, 160)               0
# _________________________________________________________________
# dense (Dense)                (None, 32)                5152
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                330
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 33
# =================================================================
# Total params: 5,976
# Trainable params: 5,976
# Non-trainable params: 0
# _________________________________________________________________









# 입력 형태

# 모양이 있는 4+D 텐서: batch_shape + (channels, rows, cols)if data_format='channels_first' 
# 또는 모양이 있는 4+D 텐서: batch_shape + (rows, cols, channels)if data_format='channels_last'.

# 출력 형태

# 모양이 있는 4+D 텐서: batch_shape + (filters, new_rows, new_cols)if data_format='channels_first'
# 또는 모양이 있는 4+D 텐서: batch_shape + (new_rows, new_cols, filters)if data_format='channels_last'.
# 패딩으로 인해 값이 변경되었을 수 있습니다 rows .cols