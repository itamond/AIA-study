import tensorflow as tf
import numpy as np
print(tf.__version__)
print("hello world")

aaa = tf.constant('hello world') #상수. 바뀌지 않는 숫자,문자
print(aaa)   #Tensor("Const:0", shape=(), dtype=string) #텐서1에서는 프린트 찍으면 모양이 나옴
#모든 텐서1의 연산은 그래프 연산 방식이다.
#때문에 하나의 과정이 더 필요
# sess = tf.Session() #세션 생성 워닝떠서 아래꺼 씀
sess = tf.compat.v1.Session() #세션 생성

print(sess.run(aaa)) #b'hello world' b는 바이너리, 그냥 붙어서 나옴