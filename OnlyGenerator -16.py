import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from random import randint

alplr = 0.2

CHANNEL = 3
HEIGHT = 64
WIDTH = 64

X_dim = HEIGHT * WIDTH * CHANNEL
z_dim = 100
h_dim = 128

numberOfMorphRows = 2
numberOfMorphColumns = 8

file = open("log.txt",'a')
file.write('-----Starting Program------')
file.flush()
data_directory = "./ourDataset/3emerald"
filelist = []
for s in os.listdir(data_directory):
    if ".jpg" in s:
        filelist.append(data_directory + "/" + s)


def data(index):
    image_contents = tf.read_file(filelist[index])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [WIDTH, HEIGHT])
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_brightness(image, max_delta=0.1)
    #image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


allImagesList = []
numberOfImages = len(filelist)
for i in range(numberOfImages):
    allImagesList.append(tf.reshape(data(i), [-1, X_dim]))

print("#images: ", numberOfImages)
current = 0


def plot(samples):
    fig = plt.figure(figsize=(40, 40))
    gs = gridspec.GridSpec(numberOfMorphRows, numberOfMorphColumns)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(WIDTH, HEIGHT, CHANNEL))

    return fig


# normalisiert erstellte Matrizen; besser als 0 - Matrizen
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


samples_z = []
for i in range(numberOfImages):
    samples_z.append(sample_z(1, z_dim))


# leaky Relu
def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)





def getlastmodel():
    iterat = 0
    for st in os.listdir("./modelsG"):
        newstring = st
        while "." in newstring:
            newstring = newstring[:-1]
        if "point" not in newstring:
            if int(newstring[6:]) > iterat:
                iterat = int(newstring[6:])

    return "./modelsG/model_%s.ckpt" % iterat, iterat


with tf.name_scope('model1'):
    # generator variabeln

    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

    theta_G = [G_W1, G_W2, G_b1, G_b2]

    # generator

    keepProb = tf.placeholder(tf.float32)


    def g_sample(i):
        vector = tf.cast(samples_z[i], tf.float32)
        g_h1 = lrelu(tf.matmul(vector, G_W1) + G_b1, alplr)
        g_h1drop = tf.nn.dropout(g_h1, keepProb)  # drop beim Testen und nihct
        g_log_prob = tf.matmul(g_h1drop, G_W2) + G_b2

        # dropout Layer

        return tf.nn.sigmoid(g_log_prob)


G_loss = []
for i in range(numberOfImages):
    G_loss.append(tf.square(allImagesList[i] - g_sample(i)))
with tf.name_scope('train'):

    G_loss[current] = tf.square(allImagesList[current]-g_sample(current))
    G_loss = tf.reduce_sum(G_loss)

    G_solver = (tf.train.AdamOptimizer().minimize(G_loss))


def morph(first, second, samplesList):
    for i in range(numberOfMorphColumns):
        morphVector = (g_sample(first) * i / (numberOfMorphColumns-1) +
                       g_sample(second) * (1 - i / (numberOfMorphColumns-1)))
        samplesList.append(sess.run(morphVector, feed_dict={keepProb: 1.0}))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
with sess.as_default():
    # Create folders we're going to use:
    if not os.path.exists('outG/'):
        os.makedirs('outG/')
    if not os.path.exists("./modelsG"):
        os.makedirs("./modelsG")
    # A Saver to save our model:
    saver = tf.train.Saver()
    # Reloading Model:
    model, iterationcounter = getlastmodel()
    if len(os.listdir("./modelsG")) > 0:
        saver.restore(sess, model)
        print("Model restored.", )
        i = iterationcounter
        print(i)
    else:
        iterationcounter = 0

for it in range(1000000):
    for i in range(numberOfImages):
        current = i
        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={keepProb: 1.0}
        )
    if it % 1 == 0 and it != 0:
        iterationcounter += 1
        log = ('Iter: {};  G_loss: {:.4}'.format(str(iterationcounter), G_loss_curr))
        print(log)
        file.write(log)
        file.flush()
        samples = []
        for i in range(numberOfMorphRows):
            rand1 = randint(0,numberOfImages-1)
            rand2 = randint(0,numberOfImages-1)
            morph(rand1, rand2, samples)

        fig = plot(samples)
        plt.savefig('outG/{}.png'.format(str(iterationcounter).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        if it %5 == 0:
            save_path = saver.save(sess, "./modelsG/model_%s.ckpt" % iterationcounter)
            print("Model saved in file: %s" % save_path)
