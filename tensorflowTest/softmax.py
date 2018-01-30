import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


sess = tf.InteractiveSession()  # 创建TensorFlow连接session
x = tf.placeholder(tf.float32, [None, 784])  # 创建placeholder,输入数据的地方
W = tf.Variable(tf.zeros([784, 10]))  # 模型每个节点的权重值
b = tf.Variable(tf.zeros([10]))  # 模型最后加权的常量
y = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax 最大概率函数，tf.nn包含了大量的神经网络的组件，tf.matmul是TensorFlow中的矩阵乘法函数

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
