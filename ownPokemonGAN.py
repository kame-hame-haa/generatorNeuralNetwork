import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

CHANNEL = 3 #1 bei grauer Farbe
HEIGHT = 28
WIDTH = 28
batch_size = 32
image_dim = HEIGHT * WIDTH * CHANNEL
z_dim = 10
h_dim = 128

data_directory = "ourDataset/all"


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



def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, stripe, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                       num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                          name=name + '_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

    # setup the convolutional layer operation
    conv1 = tf.nn.conv2d(input_data, weights, [1, stripe[0], stripe[1], 1], padding='SAME')

    # add the bias
    conv1 += bias

    # apply a ReLU non-linear activation

    conv1 = lrelu(conv1, alplr)

    return conv1

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
#vermeidet das die Matrix mit null initialisiert wird und macht eine Normalverteilung
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Input for the Generator
def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# The discriminator
def discriminator(x):
    x = tf.reshape(x,[-1,image_dim])
    d_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(d_h1, D_W2) + D_b2
    return out

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

    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, image_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[image_dim]))

    theta_G = [G_W1, G_W2, G_b1, G_b2]

    # discriminator variabeln

    real_images = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')

    D_W1 = tf.Variable(xavier_init([image_dim, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_b1, D_b2]

    # generator

    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_sample = tf.nn.sigmoid(G_log_prob)

    # discriminator

    D_real = discriminator(real_images)
    D_fake = discriminator(G_sample)

with tf.name_scope('train'):
    D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    G_loss = -tf.reduce_mean(D_fake)

    D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                .minimize(-D_loss, var_list=theta_D))
    G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                .minimize(G_loss, var_list=theta_G))

    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]



# Load saved model
def getlastmodel():
    iterat = 0     # Initialize the iteration we are at with 0
    for st in os.listdir("./models"):
        newstring = st
        while "." in newstring:
            newstring = newstring[:-1]
        if "point" not in newstring:
            if int(newstring[6:]) > iterat:
                iterat = int(newstring[6:]) 	# set the number after "model_" as our iteration

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
        print("Model restored.",)
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
            feed_dict={real_images: train_image, z: sample_z(batch_size, z_dim)}
        )

    # G
    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(batch_size, z_dim)}
    )


    if it % 100 == 0 and it != 0:
        iterationcounter += 100
        #Print current Loss
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(iterationcounter, D_loss_curr, G_loss_curr))

        if it % 1000 == 0:
            # Draw samples
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(iterationcounter).zfill(3)), bbox_inches='tight')
            plt.close(fig)

            #Save Model
            save_path = saver.save(sess, "./models/model_%s.ckpt" % iterationcounter)
print("Model saved in file: %s" % save_path)
