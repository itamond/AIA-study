from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
tf.compat.v1.set_random_seed(337)
import numpy as np
x_data = np.array([[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]])
y_data = np.array([[0], [0], [0], [1], [1], [1]])

# [실습] 시그모이드 빼고 만들기
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_data.shape[1], 1]))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]))



# hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(xp, w) + b)
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
#tf.compat.v1.matmul(xp, w) + b의 값을 1/1+e^-x sigmoid 함수에 통과 시킨다

# 3-1. 컴파일
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yp, logits=hypothesis))
#loss = 'binary_crossentropy'
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis)) #y의 값에 따라 앞이나 뒤 수식 둘 중 하나만 실행된다.

optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 3001
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
    if step<20:
        print(loss_val)
        
# 4. 평가, 예측
xp2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y_pred = tf.nn.sigmoid(tf.compat.v1.matmul(xp2, w_val) + b_val)
y_pred = tf.cast(y_pred>0.5, dtype=tf.float32)
y_predict = sess.run(y_pred, feed_dict={xp2:x_data})
print(y_predict)
print(type(y_predict))




print('acc : ', accuracy_score(y_data, y_predict))
print('mse : ', mean_squared_error(y_data, y_predict))