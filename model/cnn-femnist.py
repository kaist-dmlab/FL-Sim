import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

# https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py
class Model(AbstractModel):
    
    def createModel(self):
        conv1 = tf.layers.conv2d(
          inputs=self.x,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=self.numClasses)
        
        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=logits)
        y_hat = tf.cast(tf.argmax(logits, 1), tf.int32)
        return loss, y_hat