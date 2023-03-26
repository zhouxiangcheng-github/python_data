# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 14:37
# @File    : tensorflow_21.py

'''
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist_data_folder="/MNIST_data"	#指定数据集所在的位置（见上图存放格式）
mnist=input_data.read_data_sets(mnist_data_folder,one_hot=True)	#读取mnist数据集，指定标签格式one_hot=True

#获取数据集的个数
train_nums=mnist.train.num_examples
validation_nums=mnist.validation.num_examples
test_nums=mnist.test.num_examples
print("MNIST训练数据集个数 %d"%train_nums)
print("MNIST验证数据集个数 %d"%validation_nums)
print("MNIST测试数据集个数 %d"%test_nums)

#获取数据值
train_data=mnist.train.images   #所有训练数据
val_data=mnist.validation.images    #(5000,784)
test_data=mnist.test.images
print("训练集数据大小：",train_data.shape)
print("一幅图像大小：",train_data[1].shape)
print("一幅图像的列表表示：\n",train_data[1])

#获取标签值
train_labels=mnist.train.labels     #(55000,10)
val_labels=mnist.validation.labels  #(5000,10)
test_labels=mnist.test.labels   #(10000,10)
print("训练集标签数组大小L: ",train_labels.shape)
print("一幅图像的标签大小: ",train_labels[1].shape)
print("一幅图像的标签值：",train_labels[1])

#批量获取数据和标签  使用 next_batch(batch_size)
#注意使用改方式时数据是随机读取的，但在同一批次中，数据和标签位置是对应的
batch_size=100  #每次批量训练100幅图像
batch_xs,batch_ys=mnist.train.next_batch(batch_size)
testbatch_xs,testbatch_ys=mnist.test.next_batch(batch_size)
print("使用mnist.train.next_batch(batch_size)批量读取样本")
print("批量随机读取100个样本，数据集大小= ",batch_xs.shape)
print("批量随机读取100个样本，标签集大小= ",batch_ys.shape)
print("批量随机读取100个测试样本，数据集大小= ",testbatch_xs.shape)
print("批量随机读取100个测试样本，标签集大小= ",testbatch_ys.shape)

#显示图像
plt.figure()
for i in range(10):
    im=train_data[i].reshape(28,28)	#训练数据集的第i张图，将其转化为28x28格式
    #im=batch_xs[i].reshape(28,28)	#该批次的第i张图
    plt.imshow(im)
    plt.pause(0.1)	#暂停时间
plt.show()


'''
#import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

mnist_data_folder="/MNIST_data"
mnist=input_data.read_data_sets(mnist_data_folder,one_hot=True)


#创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#权重初始化函数
def weight_variable(shape):
    #输出服从截尾正态分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏置初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#创建卷积op
#x 是一个4维张量，shape为[batch,height,width,channels]
#卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

#创建池化op
#采用最大池化，也就是取窗口中的最大值作为结果
#x 是一个4维张量，shape为[batch,height,width,channels]
#ksize表示pool窗口大小为2x2,也就是高2，宽2
#strides，表示在height和width维度上的步长都为2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME")

#第1层，卷积层
#初始化W为[5,5,1,32]的张量，表示卷积核大小为5*5，第一层网络的输入和输出神经元个数分别为1和32
W_conv1 = weight_variable([5,5,1,32])
#初始化b为[32],即输出大小
b_conv1 = bias_variable([32])

#把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]
#-1表示自动推测这个维度的size
x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
#h_pool1的输出即为第一层网络输出，shape为[batch,14,14,1]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第2层，卷积层
#卷积核大小依然是5*5，这层的输入和输出神经元个数为32和64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = weight_variable([64])

#h_pool2即为第二层网络输出，shape为[batch,7,7,1]
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第3层, 全连接层
#这层是拥有1024个神经元的全连接层
#W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

#计算前需要把第2层的输出reshape成[batch, 7*7*64]的张量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout层
#为了减少过拟合，在输出层前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
#最后，添加一个softmax层
#可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#预测值和真实值之间的交叉墒
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

#train op, 使用ADAM优化器来做梯度下降。学习率为0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。
#因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

#计算正确预测项的比例，因为tf.equal返回的是布尔值，
#使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))


train_data=mnist.train.images
train_labels=mnist.train.labels
test_data=mnist.test.images
test_labels=mnist.test.labels

batch_size=100  #每次批量训练100幅图像
batch_xs,batch_ys=mnist.train.next_batch(batch_size)    #随机抓取训练数据中的100个批处理数据点
test_xs,test_ys=mnist.test.next_batch(batch_size)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #初始化变量
    for i in range(2000):  #开始训练模型，循环2000次，每次传入一张图像
        sess.run(train_step,feed_dict={x:[train_data[i]], y_:[train_labels[i]], keep_prob:0.5})
        if(i%100==0):   #每100次，传入一个批次的测试数据，计算其正确率
            print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0}))

"""
也可以批量导入训练，注意使用mnist.train.next_batch(batch_size)，得到的批次数据每次都会自动随机抽取这个批次大小的数据
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #初始化变量
    for i in range(200):  #开始训练模型，循环200次，每次传入一个批次的图像
        sess.run(train_step,feed_dict={x:batch_xs, y_:batch_ys, keep_prob:0.5})
        if(i%20==0):   #每20次，传入一个批次的测试数据，计算其正确率
            print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0}))
"""
