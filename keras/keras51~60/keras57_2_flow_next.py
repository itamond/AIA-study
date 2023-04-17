#수치로 제공된 데이터의 증폭

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 100   #증강, 증폭

print(x_train.shape)   #(60000, 28, 28)
print(x_train[0].shape)  #(28, 28)
print(x_train[0][0].shape)  #(28,)
print(x_train[1][0].shape)  #(28,)


print(np.tile(x_train[0].reshape(28*28),  #증폭시키기 위해서 쉐이프 바꿔줌
              augment_size).reshape(-1,28,28,1).shape)        #계속 복붙시키는 개념?
#x 트레인의 0번째를 augment_size 만큼 증폭해서 (-1,28,28,1) 모양으로 리쉐이프 해라
#np.tile(데이터, 증폭시킬갯수)

print(np.zeros(augment_size))  # zeros = 0값 출력
print(np.zeros(augment_size).shape) #(100,)
#flow_from_directory는 폴더에서 가져와서 x,y만드는것
#flow는 원래 있는 데이터 셋에서 x,y 만드는것
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),
            augment_size).reshape(-1,28,28,1),# x데이터
    np.zeros(augment_size), #y데이터 : 그림만 그릴꺼라 필요없어서 걍 0 넣음                        
    batch_size=augment_size,
    shuffle=True,
).next()                #위 코드블록을 한번 실행 시켜주는 것(이터레이터의 첫번째 배치)

 #x와 y가 다 나옴. 첫번째 


################# .next() 사용 ##############################
print(x_data)                 # 넥스트 미사용시 print(x_data[0])와 같음. x_data의 next값이기 때문
print(type(x_data))           #<class 'tuple'>
print(x_data[0])              #x데이터
print(x_data[1])              #y데이터
print(x_data[0].shape, x_data[1].shape)       #이터레이터에서 next로 호출한, 튜플 안에 넘파이가 들어가 있는 구조이기 때문에 shape를 찍을 수 있음
print(type(x_data[0]))        #<class 'numpy.ndarray'>


################## .next() 미사용 ##############################
# print(x_data)        
#<keras.preprocessing.image.NumpyArrayIterator object at 0x0000026159505EE0>
# print(x_data[0])   # x와 y가 모두 포함
# print(x_data[0][0].shape)   #(100, 28, 28, 1)
# print(x_data[0][1].shape)   #(100,)





import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)      #7바이 7의 서브플롯을 만든다.
    plt.axis('off')
    # plt.imshow(x_data[0][0][i], cmap='gray')  #.next() 미사용
    plt.imshow(x_data[0][i], cmap='gray')       #.next()를 통해서 쉐이프 하나가 줄었다. 때문에 [0] 하나 삭제
plt.show()


# 번역결과
# plt.axis()는 플롯에 대한 축의 표시 범위를 설정하는 Python의 matplotlib 라이브러리에 있는 함수입니다. 다음과 같은 몇 가지 선택적 매개변수를 사용합니다.
# xmin, xmax, ymin, ymax: 각각 가로축과 세로축의 한계. 이 중 하나라도 제공되지 않으면 matplotlib는 플로팅되는 데이터를 기반으로 자동으로 제한을 결정합니다.
# vmin, vmax: 컬러 맵을 사용하는 플롯에 대한 컬러 스케일의 한계입니다. 이것이 제공되지 않으면 matplotlib는 플롯되는 데이터를 기반으로 제한을 결정합니다.
# scalex, scaley: True인 경우, 축의 한계는 플롯되는 데이터에 따라 조정됩니다. 'False'인 경우 제한이 변경되지 않습니다.
# 종횡비: 높이와 너비의 비율로 정의되는 플롯의 종횡비입니다. 기본적으로 aspect='auto'는 matplotlib가 플롯의 크기에 따라 종횡비를 결정함을 의미합니다.
# anchor: 플롯 내 축의 위치입니다. 기본적으로 앵커는 (0, 0)에 있으며 이는 플롯의 왼쪽 아래 모서리가 축의 최소값에 해당함을 의미합니다. 다른 값에는 'C'(가운데), 'SW'(왼쪽 아래) 등이 있습니다.
# plt.axis()를 인수 없이 사용하여 현재 축 제한을 반환하거나 단일 인수를 사용하여 가로 및 세로 축의 제한을 동일한 값으로 설정할 수 있습니다.


