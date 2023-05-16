import tensorflow as tf

# x_train = [1,2,3]
# y_train = [1,2,3]
x_train = [1]
y_train = [2]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))   #mse



####################optimizer#####################
lr = 0.1

gradient = tf.reduce_mean((x * w - y) * x)     # -> 로스의 미분값이다.

# y = x*w + b
# w_f = W_i - lr * (delta e / delta w)

descent = w - lr * gradient   # descent 갱신된 웨이트
# w = descent
update = w.assign(descent)   # w = w - lr * gradient


##################################################

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    
    _, loss_v, w_v =sess.run([update, loss, w], feed_dict={x : x_train, y : y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()   #리스트는 초기화되지 않음

print("==============================w history=================================")
print(w_history)
print("==============================loss history=================================")
print(loss_history)

#체인룰 = 미분에 미분은 미분미분이다


# 시행수   로스             웨이트
# 0        378.0           [5.7999997]
# 1        107.51999       [3.5599997]
# 7        0.05694734      [1.0589159]
# 8        0.016198363     [1.0314218]
# 9        0.004607539     [1.0167583]
# 10       0.0013105891    [1.0089378]
# 11       0.00037279623   [1.0047668]
# 12       0.00010603762   [1.0025423]
# 13       3.0161003e-05   [1.0013559]
# 14       8.579332e-06    [1.0007231]
# 15       2.440236e-06    [1.0003856]
# 16       6.9393377e-07   [1.0002056]
# 17       1.9738451e-07   [1.0001097]
# 18       5.6130983e-08   [1.0000585]
# 19       1.5973896e-08   [1.0000312]
# 20       4.552286e-09    [1.0000167]
