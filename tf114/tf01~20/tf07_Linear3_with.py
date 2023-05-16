import tensorflow as tf
tf.set_random_seed(337)

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

w = tf.Variable(333, dtype=tf.float32)
b = tf.Variable(111, dtype=tf.float32)

#####[실습]#####

hypothesis = x * w + b

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2 훈련
with tf.compat.v1.Session() as sess :
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    #model.fit()
    epochs = 20001
    for step in range(epochs) :
        sess.run(train)
        if step %20 == 0 :
            print(step, 'loss :', sess.run(loss), 'w :', sess.run(w), 'b :', sess.run(b))

    # sess.close() 
    #with문은 자동으로 close() 해줌.
