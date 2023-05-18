import autokeras as ak
import time
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist
import tensorflow as tf


#1. 데이터
(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data()


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

# 결과 : [0.032291725277900696, 0.9908000230789185]
# 걸린시간 : 44.3636
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 28, 28)]          0

#  cast_to_float32 (CastToFloa  (None, 28, 28)           0
#  t32)

#  expand_last_dim (ExpandLast  (None, 28, 28, 1)        0
#  Dim)

#  normalization (Normalizatio  (None, 28, 28, 1)        3
#  n)

#  conv2d (Conv2D)             (None, 26, 26, 32)        320

#  conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496

#  max_pooling2d (MaxPooling2D  (None, 12, 12, 64)       0
#  )

#  dropout (Dropout)           (None, 12, 12, 64)        0

#  flatten (Flatten)           (None, 9216)              0

#  dropout_1 (Dropout)         (None, 9216)              0

#  dense (Dense)               (None, 10)                92170

#  classification_head_1 (Soft  (None, 10)               0
#  max)

# =================================================================
# Total params: 110,989
# Trainable params: 110,986
# Non-trainable params: 3
# _________________________________________________________________