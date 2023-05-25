from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2
from tensorflow.keras.applications import ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet121, DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

# model_list = [VGG16, VGG19, ...]

model_list = [VGG16, VGG19 , ResNet50 , ResNet50V2 , ResNet101 , ResNet101V2 , ResNet152 , ResNet152V2 , DenseNet201 , DenseNet121 , DenseNet169 , InceptionV3 , InceptionResNetV2 ,
              MobileNet , MobileNetV2 , MobileNetV3Large , MobileNetV3Small , NASNetMobile , NASNetLarge , EfficientNetB0 , EfficientNetB1 , EfficientNetB7 , Xception ]



for i in range(len(model_list)):
    model = model_list[i]()
    model.trainable = False
    print('====================================')
    print('모델명 : ', model_list[i].__name__)
    print('전체 가중치 갯수 : ', len(model.weights))
    print('훈련 가능한 가중치 갯수 : ', len(model.trainable_weights))

# model = VGG16()
# model = VGG19()

# model.summary()

###################### 결과 출력 #########################
######### for문 사용
# ====================================
# 모델명 :  VGG16
# 전체 가중치 갯수 :  32
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  VGG19
# 전체 가중치 갯수 :  38
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  ResNet50
# 전체 가중치 갯수 :  320
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  ResNet50V2
# 전체 가중치 갯수 :  272
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  ResNet101
# 전체 가중치 갯수 :  626
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  ResNet101V2
# 전체 가중치 갯수 :  544
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  ResNet152
# 전체 가중치 갯수 :  932
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  ResNet152V2
# 전체 가중치 갯수 :  816
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  DenseNet201
# ====================================
# 모델명 :  MobileNetV3Small
# 전체 가중치 갯수 :  210
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  NASNetMobile
# 전체 가중치 갯수 :  1126
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  NASNetLarge
# 전체 가중치 갯수 :  1546
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  EfficientNetB0
# 전체 가중치 갯수 :  314
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  EfficientNetB1
# 전체 가중치 갯수 :  442
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  EfficientNetB7
# 전체 가중치 갯수 :  1040
# 훈련 가능한 가중치 갯수 :  0
# ====================================
# 모델명 :  Xception
# 전체 가중치 갯수 :  236
# 훈련 가능한 가중치 갯수 :  0
