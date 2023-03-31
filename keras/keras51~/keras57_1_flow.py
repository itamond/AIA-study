#수치로 제공된 데이터의 증폭

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
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
    np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),# x데이터
    np.zeros(augment_size), #y데이터 : 그림만 그릴꺼라 필요없어서 걍 0 넣음                        
    batch_size=augment_size,
    shuffle=True,
    )


print(x_data)
#<keras.preprocessing.image.NumpyArrayIterator object at 0x0000026159505EE0>
print(x_data[0])   # x와 y가 모두 포함
print(x_data[0][0].shape)   #(100, 28, 28, 1)
print(x_data[0][1].shape)   #(100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)      #7바이 7의 서브플롯을 만든다.
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()




