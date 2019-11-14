import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取数据集
mnist = input_data.read_data_sets('E:/dataset/MNIST_data', one_hot=True)
print(mnist.test.labels.shape)

# 取一小撮数据
Batch_size = 200
xs, ys = mnist.train.next_batch(Batch_size)
print(xs.shape)
print(ys.shape)

# 保存模型
saver = tf.train.Saver()
sess = tf.Session()
saver.save(sess, os.path.join(
    'E:/dataset/parameters', mnist), ignore_expires=False)


# 加载模型
ckpt = tf.train.get_checkpoint_state('E:/dataset/parameters')
saver.restore(sess, ckpt.model_checkpoint_path)
