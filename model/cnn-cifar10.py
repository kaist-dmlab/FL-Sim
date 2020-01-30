import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

class Model(AbstractModel):
    
    def get_logits(self, x, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_conv5, b_conv5, W_fc1, b_fc1, W_fc2, b_fc2):
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
        h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='VALID') + b_conv5)
        
        h_conv5_flat = tf.reshape(h_conv5, [-1, 1 * 1 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
#         h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        logits = tf.matmul(h_fc1,W_fc2) + b_fc2
        return logits
    
    def createModel(self):
        W_conv1 = tf.get_variable('W_conv1', dtype=tf.float32, initializer=tf.truncated_normal(shape=[3, 3, 3, 64], stddev=5e-2))
        b_conv1 = tf.get_variable('b_conv1', dtype=tf.float32, initializer=tf.constant(0.1, shape=[64]))
        W_conv2 = tf.get_variable('W_conv2', dtype=tf.float32, initializer=tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
        b_conv2 = tf.get_variable('b_conv2', dtype=tf.float32, initializer=tf.constant(0.1, shape=[128]))
        W_conv3 = tf.get_variable('W_conv3', dtype=tf.float32, initializer=tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))
        b_conv3 = tf.get_variable('b_conv3', dtype=tf.float32, initializer=tf.constant(0.1, shape=[256]))
        W_conv4 = tf.get_variable('W_conv4', dtype=tf.float32, initializer=tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2))
        b_conv4 = tf.get_variable('b_conv4', dtype=tf.float32, initializer=tf.constant(0.1, shape=[256]))
        W_conv5 = tf.get_variable('W_conv5', dtype=tf.float32, initializer=tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2))
        b_conv5 = tf.get_variable('b_conv5', dtype=tf.float32, initializer=tf.constant(0.1, shape=[256]))
        W_fc1 = tf.Variable(tf.truncated_normal(shape=[1 * 1 * 256, 128], stddev=5e-2))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[128]))
        W_fc2 = tf.Variable(tf.truncated_normal(shape=[128, self.numClasses], stddev=5e-2))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        
        y_onehot = tf.one_hot(self.y, self.numClasses)
        logits = self.get_logits(self.x, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_conv5, b_conv5, W_fc1, b_fc1, W_fc2, b_fc2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
        y_prob = tf.nn.softmax(logits)
        y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
        return loss, y_hat