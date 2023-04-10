from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical

img_height, img_width = 28, 28

mnist_dir = 'D:/number/Mnist/'
fnt_dir = 'D:/number/Fnt/'
goodimg_dir = 'D:/number/GoodImg/'
hnd_dir = 'D:/number/Hnd2/'

Mnist_datagen = ImageDataGenerator(
    rescale=1./255,
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest',
)

goodimg_augment = 2000
Hndimg_augment = 1
fnt_augment = 10000


xy_good =train_datagen.flow_from_directory(                     #폴더에서 가져올거야~    
    'D:/number/GoodImg/',                #이미지제너레이터는 폴더별로 라벨값 부여. 때문에 분류 폴더 이전 상위폴더까지만 설정해도됨
    target_size=(28, 28),                            #이미지 데이터를 200x200으로 확대 혹은 축소해라. 사이즈를 동일하게 만들어준다.
    batch_size=60000,                                      #5장씩 잘라라
    class_mode='categorical',                               #0과 1을 찾는 mode, int형 수치화해서 만들어줌 
    color_mode='grayscale',
    shuffle=True,
)

xy_Hnd =train_datagen.flow_from_directory(                     #폴더에서 가져올거야~    
    'D:/number/Hnd2/',                #이미지제너레이터는 폴더별로 라벨값 부여. 때문에 분류 폴더 이전 상위폴더까지만 설정해도됨
    target_size=(28, 28),                            #이미지 데이터를 200x200으로 확대 혹은 축소해라. 사이즈를 동일하게 만들어준다.
    batch_size=150000,                                      #5장씩 잘라라
    class_mode='categorical',                               #0과 1을 찾는 mode, int형 수치화해서 만들어줌 
    color_mode='grayscale',
    shuffle=True,
)

xy_fnt =train_datagen.flow_from_directory(                     #폴더에서 가져올거야~    
    'D:/number/fnt/',                #이미지제너레이터는 폴더별로 라벨값 부여. 때문에 분류 폴더 이전 상위폴더까지만 설정해도됨
    target_size=(28, 28),                            #이미지 데이터를 200x200으로 확대 혹은 축소해라. 사이즈를 동일하게 만들어준다.
    batch_size=60000,                                      #5장씩 잘라라
    class_mode='categorical',                               #0과 1을 찾는 mode, int형 수치화해서 만들어줌 
    color_mode='grayscale',
    shuffle=True,
)

xy_mnist = Mnist_datagen.flow_from_directory(                     #폴더에서 가져올거야~    
    'D:/number/Mnist/',                #이미지제너레이터는 폴더별로 라벨값 부여. 때문에 분류 폴더 이전 상위폴더까지만 설정해도됨
    target_size=(28, 28),                            #이미지 데이터를 200x200으로 확대 혹은 축소해라. 사이즈를 동일하게 만들어준다.
    batch_size=60000,                                      #5장씩 잘라라
    class_mode='categorical',                               #0과 1을 찾는 mode, int형 수치화해서 만들어줌 
    color_mode='grayscale',
    shuffle=True,
)


x_good = xy_good[0][0]
y_good = xy_good[0][1]
x_Hnd = xy_Hnd[0][0]
y_Hnd = xy_Hnd[0][1]
x_fnt = xy_fnt[0][0]
y_fnt = xy_fnt[0][1]
x_mnist = xy_mnist[0][0]
y_mnist = xy_mnist[0][1]

# (591, 28, 28, 1) (591, 10)
# (550, 28, 28, 1) (550, 10)
# (10160, 28, 28, 1) (10160, 10)
# (60000, 28, 28, 1) (60000, 10)
good_augment_size = 2000
Hnd_augment_size = 150000
fnt_augment_size = 10000
good_randidx=np.random.randint(x_good.shape[0], size=good_augment_size)
Hnd_randidx=np.random.randint(x_Hnd.shape[0], size=Hnd_augment_size)
fnt_randidx=np.random.randint(x_fnt.shape[0], size=fnt_augment_size)

x_good_augmented = x_good[good_randidx].copy()
y_good_augmented = y_good[good_randidx].copy()
x_Hnd_augmented = x_Hnd[Hnd_randidx].copy()
y_Hnd_augmented = y_Hnd[Hnd_randidx].copy()
x_fnt_augmented = x_fnt[fnt_randidx].copy()
y_fnt_augmented = y_fnt[fnt_randidx].copy()


x_good_augmented = train_datagen.flow(
    x_good_augmented, y_good_augmented,
    batch_size = good_augment_size,
    shuffle=False,
).next()[0]


x_Hnd_augmented = train_datagen.flow(
    x_Hnd_augmented, y_Hnd_augmented,
    batch_size = Hnd_augment_size,
    shuffle=False,
).next()[0]

x_fnt_augmented = train_datagen.flow(
    x_fnt_augmented, y_fnt_augmented,
    batch_size = fnt_augment_size,
    shuffle=False,
).next()[0]



x_good = np.concatenate((x_good, x_good_augmented))
y_good = np.concatenate((y_good, y_good_augmented))
x_Hnd = np.concatenate((x_Hnd, x_Hnd_augmented))
y_Hnd = np.concatenate((y_Hnd, y_Hnd_augmented))
x_fnt = np.concatenate((x_fnt, x_fnt_augmented))
y_fnt = np.concatenate((y_fnt, y_fnt_augmented))


x = np.concatenate((x_mnist))
y = np.concatenate((y_mnist))

# print(x.shape, y.shape)
#(121301, 28, 28, 1) (121301, 10)

path = 'D:/number/_npy/'
np.save(path + 'pro_x.npy', arr=x)
np.save(path + 'pro_y.npy', arr=y)