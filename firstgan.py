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

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# pics
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

# normal distribution instead zeros
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
    

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
    
# two layer
def discriminator(x, D_W1, D_W2, D_b1, D_b2):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out

with tf.name_scope('model'):

    # generator variabels
    
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

    theta_G = [G_W1, G_W2, G_b1, G_b2]    

    # discriminator variabels

    X = tf.placeholder(tf.float32, shape=[None, X_dim])

    D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_b1, D_b2]
    
    #generator 
    

    keepProb = tf.placeholder(tf.float32)
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h1Drop = tf.nn.dropout(G_h1, keepProb) # dropout
    G_log_prob = tf.matmul(G_h1Drop, G_W2) + G_b2
        
    #dropout Layer
        

    G_sample = tf.nn.sigmoid(G_log_prob)
    
    
    
    # discriminator
   
    D_real = discriminator(X,  D_W1, D_W2, D_b1, D_b2) #network with real iMAGES
    D_fake = discriminator(G_sample,  D_W1, D_W2, D_b1, D_b2) # network with fake ones


with tf.name_scope('train'):
    # loss function

    D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    G_loss = -tf.reduce_mean(D_fake)

    #tensorboard stuff
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)

    D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                .minimize(-D_loss, var_list=theta_D))
    G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                .minimize(G_loss, var_list=theta_G))

    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tfb_merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/', sess.graph)

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    for _ in range(5):
        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr, summary = sess.run(
            [D_solver, D_loss, tfb_merged],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim), keepProb: 1.0} # training discriminator
        )
        train_writer.add_summary(summary, it)

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim), keepProb: .7} #prevents overfitting; training generator
    )
    train_writer.add_summary(summary, it)
    if it % 100 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))
        tf.summary.merge_all()
        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim), keepProb: 1.0})
            

            fig = plot(samples)
            plt.savefig('out/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
plt.close(fig)
