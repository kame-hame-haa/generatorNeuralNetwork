import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

CHANNEL = 3  # 1 bei grauer Farbe
HEIGHT = 64
WIDTH = 64
batch_size = 32
image_dim = HEIGHT * WIDTH * CHANNEL
noise_dim = 100
h_dim = 128

data_directory = "./ourDataset/all"

# Reading in the pictures
def process_data():
    images = []
    for each in os.listdir(data_directory):
        if ".jpg" in each:
            images.append(os.path.join(data_directory, each))
    all_images = tf.convert_to_tensor(images, dtype=tf.string)

    images_queue = tf.train.slice_input_producer(
        [all_images])

    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels=CHANNEL)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT, WIDTH, CHANNEL])

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    images_batch = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=200 + 3 * batch_size,
        min_after_dequeue=200)
    num_images = len(images)

    return images_batch, num_images


# drawing the generated images
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
        if CHANNEL == 1:
            plt.imshow(sample.reshape(WIDTH, HEIGHT), cmap='Greys_r', interpolation="none")
        else:
            plt.imshow(sample.reshape(WIDTH, HEIGHT, CHANNEL), interpolation="none")

    return fig


# normalisiert erstellte Matrizen; besser als 0 - Matrizen
# vermeidet das die Matrix mit null initialisiert wird und macht eine Normalverteilung
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Input for the Generator
def noise(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# leaky Relu
def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)

#Source : https://github.com/llSourcell/Pokemon_GAN/blob/master/pokeGAN.py
def generator(input, random_dim,  reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32  # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
        # Convolution, bias, activation, repeat!
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=True, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        # Convolution, bias, activation, repeat!
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=True, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=True, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=True, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*3
        conv5 = tf.layers.conv2d_transpose(act4, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        #bn5 = tf.contrib.layers.batch_norm(conv5, is_training=True, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.sigmoid(conv5, name='act5')
        return act5



#Source : https://github.com/llSourcell/Pokemon_GAN/blob/master/pokeGAN.py
def discriminator(input, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Reshape
        input = tf.reshape(input,[-1,WIDTH,HEIGHT,CHANNEL])
        # Convolution, activation, bias, repeat!
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=True, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn1')
        act1 = lrelu(bn1, n='act1')
        # Convolution, activation, bias, repeat!
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=True, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        # Convolution, activation, bias, repeat!
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=True, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
        # Convolution, activation, bias, repeat!
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=True, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')

        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')

        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        acted_out = tf.nn.sigmoid(logits)
        return logits  # , acted_out


# Initialize weights

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # normalverteilung
    return tf.Variable(initial)


# initialize biases
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


with tf.name_scope('model1'):
    # generator variabeln

    rand_input = tf.placeholder(tf.float32, shape=[None, noise_dim])



    real_images = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')

    # generator

    G_sample = generator(rand_input,noise_dim)

    # discriminator
    D_real = discriminator(real_images)
    D_fake = discriminator(G_sample, reuse=True)


with tf.name_scope('train'):
    # Loss function
    D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    G_loss = -tf.reduce_mean(D_fake)

    # discriminator variabeln
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]

    D_solver = (tf.train.RMSPropOptimizer(learning_rate=2e-4)
                .minimize(-D_loss, var_list=d_vars))
    G_solver = (tf.train.RMSPropOptimizer(learning_rate=2e-4)
                .minimize(G_loss, var_list=g_vars))

    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]


# Load saved model
def getlastmodel():
    iterat = 0  # Initialize the iteration we are at with 0
    for st in os.listdir("./models"):
        newstring = st
        while "." in newstring:
            newstring = newstring[:-1]
        if "point" not in newstring:
            if int(newstring[6:]) > iterat:
                iterat = int(newstring[6:])  # set the number after "model_" as our iteration

    return "./models/model_%s.ckpt" % iterat, iterat


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
        print("Model restored.")
        i = iterationcounter
        print(i)
    else:
        iterationcounter = 0

# Initialise batch, coordinator and thread that feed the session with the images
image_batch, samples_num = process_data()
print(samples_num)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# The main loop:
for it in range(10000000):
    # Train Discriminator five times as much as the Generator
    for _ in range(5):
        train_image = sess.run(image_batch)

        # D
        _, D_loss_curr, _ = sess.run(
            [D_solver, D_loss, clip_D],
            feed_dict={real_images: train_image, rand_input: noise(batch_size, noise_dim)}
        )

    # G
    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={rand_input: noise(batch_size, noise_dim)}
    )

    if it % 100 == 0 and it != 0:
        iterationcounter += 100
        # Print current Loss
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(iterationcounter, D_loss_curr, G_loss_curr))

        if it % 100 == 0:
            # Draw samples
            samples = sess.run(G_sample, feed_dict={rand_input: noise(16, noise_dim)})
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(iterationcounter).zfill(3)), bbox_inches='tight')
            plt.close(fig)

            # Save Model
            save_path = saver.save(sess, "./models/model_%s.ckpt" % iterationcounter)
            print("Model saved in file: %s" % save_path)
