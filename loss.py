import tensorflow as tf

#定义待优化的参数W
w=tf.Variable(tf.constant(5,dtype=tf.float32))

#定义损失函数
loss=tf.square(w+1)

#指数衰减学习率
LEARNING_RATE_BASE=0.1#最初学习率
LEARNING_RATE_DECAY=0.99#学习率衰减率
LEARNING_RATE_STEP=1#喂入多少轮Batch_size后，更新学习率，一般设为：总样本数/Batch_size
global_step=tf.Variable(0,trainable=False)#运行了几轮Batch_size的计数器，设为不被训练

learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)


#定义反向传播算法
#常规学习率learning_rate设定
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)



#生成会话
sess=tf.Session()
init_op=tf.global_variables_initializer()
sess.run(init_op)

#开始训练
for i in range(40):
    sess.run(train_step)
    global_step_val=sess.run(global_step)
    learning_rate_val=sess.run(learning_rate)
    loss_val=sess.run(loss)
    print("After %s steps:global_step is %f,learning_rate is %f,loss is %f."%(i,global_step_val,learning_rate_val,loss_val))



