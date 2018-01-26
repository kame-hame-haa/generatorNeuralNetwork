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
    figure = plt.figure(figsize=(4, 4))
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

    return figure


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

def generator(input_, random_dim,  reuse=False):
    c4, c8, c16 = 512, 256, 128  # channel num
    s4 = 8
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input_, w1), b1, name='flat_conv1')
        # 8*8*512
        # Convolution, bias, activation, repeat!
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=True, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 16*16*256
        act2 = generatorLayer(act1,c8,2)
        # 32*32*128
        act3 = generatorLayer(act2, c16, 3)
        # 64*64*3
        act4 = generatorLayer(act3, output_dim, 4)
        return act4

# Convolution, bias, activation
def generatorLayer(act, channels, number):
    conv = tf.layers.conv2d_transpose(act, channels, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                       name=('gen_conv'+str(number)))
    bn = tf.contrib.layers.batch_norm(conv, is_training=True, epsilon=1e-5, decay=0.9,
                                       updates_collections=None, scope=('bn'+str(number)))
    act = tf.nn.relu(bn, name=('act'+str(number)))
    return act


#Source : https://github.com/llSourcell/Pokemon_GAN/blob/master/pokeGAN.py
def discriminator(input_, reuse=False):
    c2, c4, c8 = 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Reshape
        input_ = tf.reshape(input_,[-1,WIDTH,HEIGHT,CHANNEL])
        # Layer 1
        act1 = discLayer(input_,c2,1)
        # Layer 2
        act2 = discLayer(act1,c4,2)
        # Layer 3
        act3 = discLayer(act2,c8,3)

        # start from act3
        dim = int(np.prod(act3.get_shape()[1:]))
        fc1 = tf.reshape(act3, shape=[-1, dim], name='fc1')

        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        return logits


# Convolution, activation, bias
def discLayer(inp,channels,number):
    conv = tf.layers.conv2d(inp, channels, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                             name=('dis_conv'+str(number)))
    bn = tf.contrib.layers.batch_norm(conv, is_training=True, epsilon=1e-5, decay=0.9,
                                       updates_collections=None, scope=('bn'+str(number)))
    act = lrelu(bn, n=('act'+str(number)))
    return act


with tf.name_scope('model1'):
    # Placeholder for noise input for generator
    rand_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
    # Placeholder for our actual images
    real_images = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')

    # Generator
    G_sample = generator(rand_input,noise_dim)

    # Discriminator
    D_real = discriminator(real_images)
    D_fake = discriminator(G_sample, reuse=True)


with tf.name_scope('train'):
    # Loss function
    D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    tf.summary.scalar('D_loss', D_loss)
    G_loss = -tf.reduce_mean(D_fake)
    tf.summary.scalar('G_loss', G_loss)
    # The variables that need to be trained
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]

    # The actual training
    D_solver = (tf.train.RMSPropOptimizer(learning_rate=2e-4)
                .minimize(-D_loss, var_list=d_vars)) # RMSProp ist besser geignet fuer batches
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

#Starting Session
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
        print(iterationcounter)
    else:
        iterationcounter = 0
    baseit = iterationcounter

# Initialise batch, coordinator and thread that feed the session with the images
image_batch, samples_num = process_data()
print(samples_num)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
tfb_merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/', sess.graph)

# The main loop:
for it in range(10000000):
    
    # Train Discriminator five times as much as the Generator
    for _ in range(5):
        train_image = sess.run(image_batch)

        # D
        _, D_loss_curr, summary = sess.run(
            [D_solver, D_loss, tfb_merged],
            feed_dict={real_images: train_image, rand_input: noise(batch_size, noise_dim)}
        )
        train_writer.add_summary(summary, baseit + it)
        

    # G
    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={rand_input: noise(batch_size, noise_dim)}
    )
    train_writer.add_summary(summary, it + baseit)

    if it % 100 == 0 and it != 0:
        iterationcounter += 100
        # Print current Loss
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(iterationcounter, D_loss_curr, G_loss_curr))

        tf.summary.merge_all()

        if it % 100 == 0:
            # Draw samples
            drawingsamples = sess.run(G_sample, feed_dict={rand_input: noise(16, noise_dim)})
            fig = plot(drawingsamples)
            plt.savefig('out/{}.png'.format(str(iterationcounter).zfill(3)), bbox_inches='tight')
            plt.close(fig)

            # Save Model
            save_path = saver.save(sess, "./models/model_%s.ckpt" % iterationcounter)
print("Model saved in file: %s" % save_path)
