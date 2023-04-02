import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator


path = "D:/study_data/_data/dog's_breed/"
save_path = "D:/study_data/_save/dog's_breed/"

stt = time.time()

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest',
)

xy_train = train_datagen.flow_from_directory(
    "D:/study_data/_data/dog's_breed/",
    target_size=(500, 500),
    batch_size=2000,
    class_mode='categorical',
    color_mode='rgba',
    shuffle=True,
)


np.save(save_path + 'dog_breed_x_train500.npy', arr=xy_train[0][0])
np.save(save_path + 'dog_breed_y_train500.npy', arr=xy_train[0][1])


ett1 = time.time()

print('이미지 수치화 소요 시간 :', np.round(ett1-stt, 2))