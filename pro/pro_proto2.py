from keras.preprocessing.image import ImageDataGenerator
import numpy as np





np.random.seed(3123)   #시드값 부여 하는법

mnist_datagen = ImageDataGenerator(rescale=1./255)

handwriting_datagen = ImageDataGenerator(rotation_range=20,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         shear_range=0.1,
                                         zoom_range=0.1,
                                         fill_mode='nearest',
                                         rescale=1./255)

font_datagen = ImageDataGenerator(rotation_range=20,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1,
                                 zoom_range=0.1,
                                 fill_mode='nearest',
                                 rescale=1./255)


batch_size = 60000
img_height, img_width = 28, 28

mnist_dir = 'C:/project_mnist/numbers/mnist_png/'
handwriting_dir = 'C:/project_mnist/numbers/chars74k_png/Hnd/'
font_dir = 'C:/project_mnist/numbers/chars74k_png/Fnt/'

mnist_generator = handwriting_datagen.flow_from_directory(mnist_dir,
                                                          target_size=(img_height, img_width),
                                                          batch_size=batch_size,
                                                          class_mode='categorical',
                                                          shuffle=True)

handwriting_generator = handwriting_datagen.flow_from_directory(handwriting_dir,
                                                                target_size=(img_height, img_width),
                                                                batch_size=batch_size,
                                                                class_mode='categorical',
                                                                shuffle=True)

font_generator = font_datagen.flow_from_directory(font_dir,
                                                  target_size=(img_height, img_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)

train_generator = zip(mnist_generator, handwriting_generator.repeat(20), font_generator)



print(train_generator.shape)