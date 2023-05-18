import autokeras as ak
import time
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist, cifar10, cifar100
import tensorflow as tf


#1. 데이터
(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.cifar100.load_data()


#2. 모델
model = ak.ImageClassifier(
    overwrite=False,
    max_trials=2
)


#3. 컴파일, 훈련

start = time.time()
model.fit(x_train,y_train, epochs=10, validation_split=0.15)
end = time.time()







#4. 평가, 예측
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('결과 :', results)
print('걸린시간 :', round(end-start, 4))

# 최적의 모델 출력
best_model = model.export_model()
print(best_model.summary())

# 최적의 모델 저장
path = './_save/autokeras/'
best_model.save(path + 'keras62_autokeras2.h5')

