import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
tf.compat.v1.set_random_seed(337)

x_data=[[1,2,1,1],
        [2,1,3,2],
        [3,1,3,4],
        [4,1,5,5],
        [1,7,5,5],
        [1,2,5,6],
        [1,6,6,6],
        [1,7,6,7],
        ]

y_data = [[0,0,1],    #2
          [0,0,1],
          [0,0,1],
          [0,1,0],    #1
          [0,1,0],
          [0,1,0],
          [1,0,0],    #0
          [1,0,0],
          ]

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.Variable(tf.random.normal([4, 3]), name='weight')
b = tf.Variable(tf.zeros([1, 3]), name='bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1 컴파일

loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)


epochs = 1000
with tf.compat.v1.Session() as sess:
    for step in range(epochs):
        sess.run(tf.compat.v1.global_variables_initializer())
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        # print(w_val[0][0])
    # 4. 평가, 예측
    xp2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
    y_pred = tf.compat.v1.matmul(xp2, w_val) + b_val
    y_predict = sess.run([y_pred], feed_dict={xp2:x_data})
    print('r2 : ', r2_score(y_data, y_predict[0]))


# r2 :  -244.58336231944563