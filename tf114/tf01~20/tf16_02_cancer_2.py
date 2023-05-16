import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
data = load_breast_cancer()

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# 모델 정의
with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, shape=[None, 30], name="input_x")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="input_y")

with tf.name_scope("hidden_layer"):
    W1 = tf.Variable(tf.random_normal([30, 64]), name="weight1")
    b1 = tf.Variable(tf.random_normal([64]), name="bias1")
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

with tf.name_scope("output_layer"):
    W2 = tf.Variable(tf.random_normal([64, 1]), name="weight2")
    b2 = tf.Variable(tf.random_normal([1]), name="bias2")
    logits = tf.matmul(h1, W2) + b2
    y_pred = tf.nn.sigmoid(logits)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

with tf.name_scope("train"):
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 학습
    for epoch in range(5000):
        _ = sess.run([train_op], feed_dict={x: x_train, y: y_train.reshape(-1,1)})

    # 예측
    y_pred_ = sess.run(y_pred, feed_dict={x: x_test})

print(y_pred)
print(y_test)

score = accuracy_score(y_test, y_pred_.round())
print("Accuracy score: {:.2f}".format(score))