import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
mnist_train_labels = mnist.train.labels.astype(np.int)
mnist_test_labels = mnist.test.labels.astype(np.int)

feature_column = [tf.contrib.layers.real_valued_column("", dimension=784)]

classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_column,
    hidden_units=[1024, 512, 256],
    n_classes=10,
    model_dir="./mnist_"
)

for _ in range(20):
    classifier.fit(x=mnist.train.images, y=mnist_train_labels, steps=100, batch_size=100)

accuracy_score = classifier.evaluate(x=mnist.test.images, y=mnist_test_labels)["accuracy"]
print(accuracy_score)
