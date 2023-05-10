import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) #즉시실행모드 확인 #tf1 = False, tf2 = True


tf.compat.v1.disable_eager_execution() #즉시실행 종료

print(tf.executing_eagerly())

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()



# placeholder = 빈 공간에 값을 받을 준비를 하는 함수. 빈 방을 만듬
# 인풋에만 사용할 수 있음.
# feed dict가 따라다님
a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)

add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4.5})) #키밸류 형태로 입력
#7.5
print(sess.run(add_node, feed_dict={a:[1,3], b:[2,4]}))
#[3. 7.]

add_and_triple = add_node * 3
print(add_and_triple) #Tensor("mul:0", dtype=float32) 그래프로 나옴

# print(sess.run(add_and_triple, feed_dict={a:[1,3], b:[2,4]}))
# [ 9. 21.]
print(sess.run(add_and_triple, feed_dict={a:7, b:3}))
