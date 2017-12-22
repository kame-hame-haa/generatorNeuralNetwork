import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 32
X_dim = 784
z_dim = 10
h_dim = 128
dropoutRate = 0.7
alplr = 0.2


NUMBER_1CNN = 64

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


# leaky Relu
def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
    

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
    

def discriminator(x, D_W1, D_W2, D_b1, D_b2):
    #D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    x_shaped = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = create_new_conv_layer(x_shaped, 1, NUMBER_1CNN, [5, 5], [2, 2], name='cnnlayer1') #Farben auch hier aendern!
    
    
    flattened = tf.reshape(conv1, [-1, 14 * 14 * NUMBER_1CNN])
    
    # setup some weights and bias values for this layer, then activate with sigmod
    
    dense_layer1 = tf.matmul(flattened, D_W1) + D_b1
    dense_layer1 = lrelu(dense_layer1, alplr)

    
    # another layer with softmax activations
    #wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
    #bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, D_W2) + D_b2
    out = lrelu(dense_layer2, alplr)

    return out

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, stripe, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    conv1 = tf.nn.conv2d(input_data, weights, [1, stripe[0], stripe[1], 1], padding='SAME')

    # add the bias
    conv1 += bias

    # apply a ReLU non-linear activation

    conv1 = lrelu(conv1, alplr)


    # now perform max pooling
   # ksize = [1, pool_shape[0], pool_shape[1], 1]
    #strides = [1, 2, 2, 1]
   # out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                              # padding='SAME')
                               
   # reducing_layer_shape = [stripe_reducing_shape[0], stripe_reducing_shape[1], 32, 1]
                               
   # out_layer = tf.nn.conv2d(conv1, weights, [1, 1, 1, 1], padding='SAME')

    return conv1    
    
    

with tf.name_scope('model1'):

    # generator variabeln
    
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

    theta_G = [G_W1, G_W2, G_b1, G_b2]    

    # discriminator variabeln

    X = tf.placeholder(tf.float32, shape=[None, X_dim])

      # wd1 = tf.Variable(tf.truncated_normal([14 * 14 * 32, 1000], stddev=0.03), name='wd1')
      # bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
    D_W1 = tf.Variable(xavier_init([14*14*NUMBER_1CNN, 1000]))
    D_b1 = tf.Variable(tf.zeros(shape=[1000]))
    
    
        #wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
        #bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')

    D_W2 = tf.Variable(xavier_init([1000, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_b1, D_b2]
    
    #generator 
    

    keepProb = tf.placeholder(tf.float32)
    G_h1 = lrelu(tf.matmul(z, G_W1) + G_b1, alplr)
    G_h1Drop = tf.nn.dropout(G_h1, keepProb) #drop beim Testen und nihct 
    G_log_prob = tf.matmul(G_h1Drop, G_W2) + G_b2
        
    #dropout Layer
        
    G_sample = tf.nn.sigmoid(G_log_prob)
    
    # discriminator
   
    D_real = discriminator(X,  D_W1, D_W2, D_b1, D_b2)
    D_fake = discriminator(G_sample,  D_W1, D_W2, D_b1, D_b2)


with tf.name_scope('train'):

    D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    G_loss = -tf.reduce_mean(D_fake)

    D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                .minimize(-D_loss, var_list=theta_D))
    G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                .minimize(G_loss, var_list=theta_G))

    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    for _ in range(5):
        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr, _ = sess.run(
            [D_solver, D_loss, clip_D],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim), keepProb: 1.0}
        )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim), keepProb: 1.0}
    )

    if it % 100 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        if it % 00 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim), keepProb: 1.0})
            #print(samples)

            fig = plot(samples)
            plt.savefig('out/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
