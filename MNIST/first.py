## importieren

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets ("MNIST_data/", one_hot=True)

import tensorflow as tf

with tf.name_scope('model'):
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

with tf.name_scope('train'):
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum (y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    for i in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        session.run(train_step, feed_dict= {x: batch_xs, y_: batch_ys })
    
    #Test
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accurancy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
