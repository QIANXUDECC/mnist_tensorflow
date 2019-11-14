import numpy as np
import tensorflow as tf

##简单的计算
sess=tf.Session()
x = tf.constant([[1.0,2.0]]) 


##参数
w1=tf.Variable(tf.random_normal([2,3],stddev=2,mean=0,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=2,mean=0,seed=1))

#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#用会话计算结果
init_op=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init_op)
print(sess.run(y))

#喂数据
#x1=tf.placeholder(tf.float32,shape=(1,2))
#sess.run(y,feed_dict={x1:[[0.5,0.6]]})

#x2=tf.placeholder(tf.float32,shape=(None,2))
#sess.run(y,feed_dict={x2:[[0.1,0.2],[0.3,0.4],[0.5,0.6]]})


