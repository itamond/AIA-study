import numpy as np
import tensorflow.compat.v1 as tf
tf.set_random_seed(337)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

xp = tf.placeholder(tf.float32, shape=[None, 2])
yp = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.zeros([1]))

#2. 모델
hypothesis = tf.sigmoid(tf.matmul(xp, w) + b)

#3-1 컴파일
cost = -tf.reduce_mean(yp*tf.log(hypothesis) + (1-yp)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, yp), dtype=tf.float32))

#tf.equal(predicted, y) 에서는 true와 false로 반환
#cast = float32 형태로 자료형 변환 false true true false => 0 1 1 0

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict = {xp:x_data, yp:y_data})
        
        if step %200 ==0:
            print(step, cost_val)
            
            
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={xp:x_data, yp:y_data})
    print("예측값 :", h, '\n 원래값 :', p, '\n Accuracy :', a)
