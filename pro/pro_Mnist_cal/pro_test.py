from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical

img_height, img_width = 28, 28

hr_dir = 'D:/number/MMP/'

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

xy_hr =test_datagen.flow_from_directory(                     #폴더에서 가져올거야~    
    'D:/number/MMP/',                #이미지제너레이터는 폴더별로 라벨값 부여. 때문에 분류 폴더 이전 상위폴더까지만 설정해도됨
    target_size=(28, 28),                            #이미지 데이터를 200x200으로 확대 혹은 축소해라. 사이즈를 동일하게 만들어준다.
    batch_size=60000,                                      #5장씩 잘라라
    class_mode='categorical',                               #0과 1을 찾는 mode, int형 수치화해서 만들어줌 
    color_mode='grayscale',
    shuffle=False,
)

x_hr = xy_hr[0][0]
y_hr = xy_hr[0][1]
# print(x_hr)
# print(y_hr)
# print(x_hr.shape)
# print(y_hr.shape)

path = 'D:/number/_npy/'
np.save(path + 'pro_x_MMP.npy', arr=x_hr)
np.save(path + 'pro_y_MMP.npy', arr=y_hr)