import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)  #텐서플로 상수형 형태 3.0이라는 값
node2 = tf.constant(4.0) #.0을 넣었으므로 이미 float 형태


# node3 = node1 + node2
node3 = tf.add(node1, node2)

#텐서1은 노드 연산 방식이다.
#노드 1과 노드2를 연결하는 노드3 모두가 하나의 텐서 머신이 되는 것

print(node3) #Tensor("add:0", shape=(), dtype=float32) 텐서 그래프의 모양이 나옴
            #node3은 sess.run 전까지 7.0의 값이 아니다. 단순 더하기 연산한다는 뜻. 때문에 add:0이라고 나옴
sess = tf.compat.v1.Session()

print(sess.run(node3))   #7.0


print(node1) #Tensor("Const:0", shape=(), dtype=float32)

print(sess.run(node1)) #3.0

