######## 현 버전이 tf 1.0이면 그냥 출력
######## 현 버전이 tf 2.0이면 즉시실행모드를 끄고 출력
######## if문 써서 1번 소스를 변경


import tensorflow as tf
print(tf.__version__)

# 즉시실행모드
# print(tf.executing_eagerly()) #텐서1 = False   텐서2 = True
# # 즉시실행모드 비활성
# tf.compat.v1.disable_eager_execution() # 텐서 2.0을 텐서1.0 방식으로 사용
# # 즉시실행모드 활성
# tf.compat.v1.enable_eager_execution() # 

# 성호's코드
# if int(tf.__version__[0])>1:
#     tf.compat.v1.disable_eager_execution()
# print(tf.executing_eagerly())


if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

aaa = tf.constant('hello world')

with tf.compat.v1.Session() as sess:
    print(sess.run(aaa))



#컨스턴트, #배리어블(변수) #플레이스홀드