import tensorflow as tf
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

CHANNEL = 3
HEIGHT = 28
WIDTH = 28

X_dim = HEIGHT * WIDTH * CHANNEL
mb_size = 32
z_dim = 10
h_dim = 128

data_directory = "./ourDataset/all"
filelist = []
for s in os.listdir(data_directory):
    if ".png" in s:
        filelist.append(data_directory + "/" + s)


def data(index):
    image_contents = tf.read_file(filelist[index])
    image = tf.image.decode_png(image_contents, channels=3)
    image = tf.image.resize_images(image, [28, 28])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image

# images = []
# for img in filelist:
#    image_contents = tf.read_file(img)
#    image = tf.image.decode_png(image_contents, channels=3)
#    image = tf.image.resize_images(image, [28, 28])
#   images.append(image)


def plot(samplefigs):
    returnfig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for x, sample in enumerate(samplefigs):
        ax = plt.subplot(gs[x])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28, 3))              # TODO: is this right?

    return returnfig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def discriminator(x, d_w1, d_w2, d_b1, d_b2):
    d_h1 = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
    out = tf.matmul(d_h1, d_w2) + d_b2
    return out


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # Normalverteilung
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def next_batch(num):
    # Return `num` random samples
    randomnumbers = []
    for _ in range(num):
        randomnumbers.append(randint(0, len(filelist) - 1))

    nextbatch = []
    for ran in randomnumbers:
        nextbatch.append(data(ran))

    return nextbatch


def getlastmodel():
    iterat = 0
    for st in os.listdir("./models"):
        newstring = st
        while "." in newstring:
            newstring = newstring[:-1]
        if "point" not in newstring:
            if int(newstring[6:]) > iterat:
                iterat = int(newstring[6:])

    return "./models/model_%s.ckpt" % iterat, iterat


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

    D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))
    
    #drop out Layer
    

    theta_D = [D_W1, D_W2, D_b1, D_b2]

    # Generator

    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)

    # discriminator

    D_real = discriminator(X, D_W1, D_W2, D_b1, D_b2)
    D_fake = discriminator(G_sample, D_W1, D_W2, D_b1, D_b2)

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
with sess.as_default():
    # Create folders we're going to use:
    if not os.path.exists('out/'):
        os.makedirs('out/')
    if not os.path.exists("./models"):
        os.makedirs("./models")
    # A Saver to save our model:
    saver = tf.train.Saver()
    # Reloading Model:
    model, iterationcounter = getlastmodel()
    if len(os.listdir("./models")) > 0:
        saver.restore(sess, model)
        print("Model restored.",)
        i = iterationcounter
        print(i)
    else:
        iterationcounter = 0

    for it in range(1000000):
        for _ in range(5):
            batch = next_batch(mb_size)
            # Reshaping from shape: (32,#imgs,28,28,3) to (#imgs,(2352)) because 2352 = 28*28*3
            xshape = X.get_shape().as_list()
            dim = np.prod(xshape[1:])
            batch_reshaped = tf.reshape(batch, [-1, dim])
            batch_eval = sess.run(batch_reshaped)   # evaluating tensor to array

            _, D_loss_curr, _ = sess.run(
                [D_solver, D_loss, clip_D],
                feed_dict={X: batch_eval, z: sample_z(mb_size, z_dim)}
            )

        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={z: sample_z(mb_size, z_dim)}
        )
        print(it)

        if it % 100 == 0 and it != 0:
            iterationcounter += 100
            print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
                  .format(str(iterationcounter), D_loss_curr, G_loss_curr))

            if it % 100 == 0:
                samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

                fig = plot(samples)
                plt.savefig('out/{}.png'
                            .format(str(iterationcounter).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                save_path = saver.save(sess, "./models/model_%s.ckpt" % iterationcounter)
                print("Model saved in file: %s" % save_path)
