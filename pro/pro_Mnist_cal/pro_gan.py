
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import Regularizer
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageFont, ImageDraw
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import PIL

#1. 데이터


x_train = np.load('D:/number/_npy/pro_x_cb.npy')
y_train = np.load('D:/number/_npy/pro_y_cb.npy')


num1 = image.load_img('D:/number/PRED/p8.png',
                      target_size=(28, 28),
                      color_mode='grayscale')
num2 = image.load_img('D:/number/PRED/p6.png',
                      target_size=(28, 28),
                      color_mode='grayscale')


datagen = ImageDataGenerator(
    rescale=1./255,
    )

num1 = image.img_to_array(num1)
num2 = image.img_to_array(num2)
num1 = np.expand_dims(num1, axis=0)
num2 = np.expand_dims(num2, axis=0)
num1 = datagen.flow(num1, batch_size=1)
num2 = datagen.flow(num2, batch_size=1)


model = Sequential()
model.add(Conv2D(32, (5,5), strides=1, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                 input_shape=(28, 28, 1)))
model.add(Conv2D(32, (5,5), strides=1, activation='relu',
                 use_bias=True,))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), strides= 1, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),))
model.add(Conv2D(64, (3,3), strides=1, use_bias=True))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, use_bias=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, use_bias=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(284, use_bias=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.load_weights('D:/number/h5/pro_cal.h5')


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])


num1 = np.argmax(model.predict(num1), axis=1)
num2 = np.argmax(model.predict(num2),axis=1)

num1 = num1[0]
num2 = num2[0]

print(num1)
print(num2)

def calculator(num1, num2, operator):
 if operator == "+":
    result = num1 + num2
 elif operator == "-":
    result = num1 - num2
 elif operator == "*":
    result = num1 * num2
 elif operator == "/":
    result = np.round(num1 / num2, 2)
 else:
    print("Invalid operator!")
    return None
 return result

operator = "/"

result = calculator(num1, num2, operator)
print('result :', result)


def save_result_image(num1, num2, operator, result):
   img = Image.new('RGB', (220, 50), color=(255, 255, 255))
   font = ImageFont.truetype("arial.ttf", 30)
   draw = ImageDraw.Draw(img)
   draw.text((10, 10), f"{num1} {operator} {num2}", font=font, fill=(0, 0, 0))
   img.save("D:/number/cal_img/cal_d.jpg")
   
   
   
save_result_image(num1,num2,operator,result)