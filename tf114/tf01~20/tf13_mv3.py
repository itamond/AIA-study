import tensorflow as tf
tf.compat.v1.set_random_seed(337)

x_data = [[73, 52, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152], [185], [180], [205], [142]]

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name = 'weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name = 'bias')

# 2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b

# x.shape = (5, 3)
# y.shape = (5, 1)
# hypothesis = (5, 1)
# (5, 3) * (?, ?) = (5, 1)
# (?, ?) = (3, 1)


# 3. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if step %20 == 0:
            print(step, loss_val, w_val, b_val)
    y_pred = tf.compat.v1.matmul(x, w_val) + b_val
    y_predict = sess.run([y_pred], feed_dict={x:x_data})
    print(y_predict)
    print(y_predict[0])
    from sklearn.metrics import r2_score, mean_squared_error
    print('r2 : ', r2_score(y_data, y_predict[0]))
    print('mse : ', mean_squared_error(y_data, y_predict[0]))
    
    rmse = tf.sqrt(mean_squared_error(y_data, y_predict[0]))
    rmse_result = sess.run(rmse)
    print('rmse : ', rmse_result)