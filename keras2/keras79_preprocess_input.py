from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
# model = ResNet50(weights=None)
# model = ResNet50(weights='경로')

# path = 'C:/AIA/dogs-vs-cats-redux-kernels-edition/train/dog.5.jpg'
path = 'C:/Users/Administrator/Desktop/나4.jpg'

img = image.load_img(path, target_size = (224, 224))
print(img)

x = image.img_to_array(img)
print('========================== image.img_to_array(img) ==========================')
# print(x, '\n', x.shape)
# print(np.min(x), np.max(x))  #0.0 255.0

# x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
x = x.reshape(1, *x.shape)
print(x.shape)  #(1, 224, 224, 3)

# x = np.expand_dims(x, axis = 0) # 축 늘리는 함수
# print(x.shape) #(1, 1, 224, 224, 3)

#####################-155에서 155 사이로 정규화######################
print('========================== preprocess_input(x) ==========================')

x = preprocess_input(x)
print(x.shape)  #(1, 224, 224, 3)

print(np.min(x), np.max(x)) 

print('========================== model.predict(x) ============================')
x_pred = model.predict(x)
print(x_pred, '\n', x_pred.shape)
print('결과는 : ', decode_predictions(x_pred, top=5)[0]) #복호화 해제 함수, 최고성능 5개를 풀어달라는 뜻