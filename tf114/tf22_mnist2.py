#pip install keras==1.2.2
#from tensorflow.keras.datasets import mnist
import tensorflow as tf
from keras.datasets import mnist
import keras
import numpy as np
from keras.utils.np_utils import to_categorical #옛날 원핫
from sklearn.metrics import accuracy_score

print(keras.__version__)
tv = tf.compat.v1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,)
# (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# (60000, 784) (60000, 10)
# (10000, 784) (10000, 10)

#2. 모델 구성
x = tv.placeholder(tf.float32, [None, 784])
y = tv.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_normal([784,32]), name='w1')   #random_uniform 균등 분포(N빵), random_normal 정규 분포 [1]은 1개짜리
b1 = tf.Variable(tf.zeros([32]), name='b1')
layer1 = tv.matmul(x, w1) + b1
dropout1 = tv.nn.dropout(layer1, rate = 0.1)

# w2 = tf.Variable(tf.random_normal([64,64]), name='w2')
# b2 = tf.Variable(tf.zeros([64]), name='b2')
# layer2 = tf.nn.relu(tv.matmul(dropout1, w2) + b2)


# w3 = tf.Variable(tf.random_normal([64,32]), name='w3')
# b3 = tf.Variable(tf.zeros([32]), name='b3')
# layer3 = tf.nn.selu(tv.matmul(layer2, w3) + b3)


w4 = tf.Variable(tf.random_normal([32,10]), name='w4')
b4 = tf.Variable(tf.zeros([10]), name='b4')
hypothesis = tv.matmul(dropout1, w4) + b4

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hypothesis))


train = tv.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

epochs = 500

sess = tv.Session()
sess.run(tv.global_variables_initializer())


for step in range(epochs) :
    # _, loss_val = sess.run([train, loss], feed_dict = {x:x_train, y:y_train})
    cost_val, _, w_val, b_val = sess.run([loss, train, w4, b4], feed_dict={x:x_train, y:y_train})
    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    print(f'epoch : {step+1}\t\t{(step+1)*100/epochs}%_complete\t\tloss : {cost_val}')
print('acc :', accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1)))