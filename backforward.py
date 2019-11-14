import numpy as np
import tensorflow as tf
BATCH_SIZE=8
seed=23455

#基于seed产生随机数
rng=np.random.RandomState(seed)
#随机生成32行2列的矩阵
X=rng.rand(32,2)
#行的和小于1，赋值1； 大于等于1，赋值0
Y=[[int(x0+x1<1)] for (x0,x1) in X]
print(X)
print(Y)

#定义数据集的输入
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

#定义参数
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))


#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#定义损失函数
loss=tf.reduce_mean(tf.square(y-y_))
#定义反向传播算法
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#生成会话
sess=tf.Session()
init_op=tf.global_variables_initializer()
sess.run(init_op)

##输出训练前，w1,w2的取值
print(sess.run(w1))
print(sess.run(w2))

##训练模型

STEPS=3000
for i in range(STEPS):
    start=(i*BATCH_SIZE)%32
    end=start+BATCH_SIZE
    sess.run(train_step,feed_dict={x:X[start:end], y_:Y[start:end]})
    if i%500==0:
        total_loss=sess.run(loss,feed_dict={x:X, y_:Y})
        print("After %d traing steps,loss on all data is %g"%(i,total_loss))

##输出训练后，w1,w2的取值
print(sess.run(w1))
print(sess.run(w2))

