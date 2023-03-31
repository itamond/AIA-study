# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition


#넘파이까지 저장
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator



path = 'd:/study_data/_data/cat_dog/Petimages/'
save_path = 'd:/study_data/_save/cat_dog/'



# np.save(path+'파일명', arr=)

stt = time.time()

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,              #MinMax스케일링(정규화) 하겠다는 의미, . 을 붙힌 이유는 부동소수점으로 연산해라 라는 뜻
    # horizontal_flip=True,        #가로 뒤집기
    # vertical_flip=True,          #수직 뒤집기
    # width_shift_range=0.1,       #10%만큼을 좌우로 움직일 수 있다는 뜻
    # height_shift_range=0.1,      #상하로 10% 움직일 수 있다는 뜻
    # rotation_range=5,            #돌릴 수 있는 범위
    # zoom_range=1.2,              #20%만큼 확대하겠다는 뜻
    # shear_range=0.7,             #찌그러트릴 수 있는 범위
    # fill_mode='nearest',         #이미지를 움직일 때, 움직여서 없어진 범위에 근처의 값을 입력해주는 기능
)                                #숫자 6하고 9같은 반전하면 데이터가 꼬이는 경우도 있다. 이럴 경우 옵션 조절해야함

test_datagen = ImageDataGenerator(
    rescale=1./255,            
    # horizontal_flip=True,       #테스트데이터는 평가하는 데이터이기때문에 데이터를 증폭한다는건 결과를 조작하는것이다. 때문에 스케일 제외한 옵션들 삭제 
    # vertical_flip=True,         #통상적으로 테스트 데이터는 증폭하지 않는다
    # width_shift_range=0.1,      
    # height_shift_range=0.1,    
    # rotation_range=5,         
    # zoom_range=1.2,           
    # shear_range=0.7,        
    # fill_mode='nearest',
)   





xy_train =train_datagen.flow_from_directory(                     #폴더에서 가져올거야~    
    'd:/study_data/_data/cat_dog/Petimages/',                #이미지제너레이터는 폴더별로 라벨값 부여. 때문에 분류 폴더 이전 상위폴더까지만 설정해도됨
    target_size=(300, 300),                            #이미지 데이터를 200x200으로 확대 혹은 축소해라. 사이즈를 동일하게 만들어준다.
    batch_size=25000,                                      #5장씩 잘라라
    class_mode='binary',                               #0과 1을 찾는 mode, int형 수치화해서 만들어줌 
    # color_mode='rgb',
    color_mode='rgb',
    shuffle=True,
)   #Found 160 images belonging to 2 classes.   0과 1의 클래스로 분류되었다.        #x=160, 200, 200, 1 로 변환 됐음  y=160,

# xy_test = test_datagen.flow_from_directory(
#     'd:/study_data/_data/brain/test/',
#     target_size=(300, 300),
#     batch_size=50000,                                      #전체 데이터를 배치로 잡아도 된다.
#     class_mode='binary',           #y의 클래스에 대한 얘기     binary=수치로 빼라는 얘기     categorical = 원핫시켜서 위치로 저장
#     color_mode='rgb',
#     shuffle=True,
# )   #Found 120 images belonging to 2 classes.   0과 1의 클래스로 분류되었다.        #x=120, 200, 200, 1 로 변환 됐음  y=120,

ett1 = time.time()
print('이미지 수치화 소요 시간 :', np.round(ett1-stt, 2))


print(xy_train[0][0])


path = 'd:/study_data/_save/_npy/'
np.save(save_path + 'keras56_x_train.npy', arr=xy_train[0][0])          #수치화된 데이터를 np형태로 저장
# np.save(path + 'keras56_x_test.npy', arr=xy_test[0][0])    
np.save(save_path + 'keras56_y_train.npy', arr=xy_train[0][1])    
# np.save(path + 'keras56_y_test.npy', arr=xy_test[0][1])


ett2 = time.time()



print('넘파이 변경 소요 시간 :', np.round(ett2-stt, 2))


