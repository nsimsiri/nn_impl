
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def naive_batch_norm(x_hat, hidden_units, epsilon):
    mean, var = tf.nn.moments(x_hat, [1])
    z_hat = tf.div(x_hat - mean, tf.sqrt(var + epsilon))
    gamma = tf.Variable(tf.ones([hidden_units]), name="Gamma")
    beta = tf.Variable(tf.zeros([hidden_units]), name="Beta")
    return gamma*z_hat + beta

def inference(images, hidden1_units, hidden2_units, epsilon=0.001, decay=0.95, batch_norm_ver=None):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    bias = tf.Variable(tf.zeros([hidden1_units]))
    x_hat = tf.matmul(images, weights) + bias

    BN_x = None
    if (batch_norm_ver == "naive"):
        BN_x = naive_batch_norm(x_hat, hidden1_units, epsilon=epsilon)
    else:
        BN_x = tf.contrib.layers.batch_norm(x_hat, decay=decay, is_training=True,
                                              updates_collections=None)
    hidden1 = tf.nn.relu(BN_x)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    bias = tf.Variable(tf.zeros([hidden1_units]))
    x_hat =tf.matmul(hidden1, weights) + bias

    BN_x = None
    if (batch_norm_ver == "naive"):
        BN_x = naive_batch_norm(x_hat, hidden2_units, epsilon=epsilon)
    else:
        BN_x = tf.contrib.layers.batch_norm(x_hat, decay=decay, is_training=True,
                                              updates_collections=None)
    hidden2 = tf.nn.relu(BN_x)
    # hidden2 = tf.nn.relu(z_hat*gamma + beta)
  # Linear
  with tf.name_scope('softmax'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    x_hat = tf.matmul(hidden2, weights) + biases
    logits = tf.nn.softmax(x_hat)
  return logits


def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
  tf.summary.scalar('loss', loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))
