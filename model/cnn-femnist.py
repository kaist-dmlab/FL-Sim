import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

class Model(AbstractModel):
    
    def get_logits_cnn(self, x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2):
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)

        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        h_conv2 = tf.nn.relu(tf.nn.conv2d(norm1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
        norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

        h_pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        h_pool2_flatten = tf.keras.layers.Flatten()(h_pool2)

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, W_fc1) + b_fc1)
        logits = tf.matmul(h_fc1,W_fc2) + b_fc2
        return logits
    
    def createModel(self):
        x_exp = tf.expand_dims(self.x, axis=-1)
        tf.set_random_seed(1234) # 모든 노드가 같은 Initial Random Seed 를 갖지 않으면 학습되지 않음 (FedAvg 논문 참조)
        W_conv1 = tf.get_variable('W_conv1', dtype=tf.float32, initializer=tf.truncated_normal(shape=[5, 5, 1, 64], stddev=5e-2))
        b_conv1 = tf.get_variable('b_conv1', dtype=tf.float32, initializer=tf.constant(0.0, shape=[64]))
        W_conv2 = tf.get_variable('W_conv2', dtype=tf.float32, initializer=tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
        b_conv2 = tf.get_variable('b_conv2', dtype=tf.float32, initializer=tf.constant(0.0, shape=[64]))
        W_fc1 = tf.get_variable('W_fc1', dtype=tf.float32, initializer=tf.truncated_normal(shape=[1024, 256], stddev=5e-2))
        b_fc1 = tf.get_variable('b_fc1', dtype=tf.float32, initializer=tf.constant(0.0, shape=[256]))
        W_fc2 = tf.get_variable('W_fc2', dtype=tf.float32, initializer=tf.truncated_normal(shape=[256, 10], stddev=5e-2))
        b_fc2 = tf.get_variable('b_fc2', dtype=tf.float32, initializer=tf.constant(0.0, shape=[10]))
        
        y_onehot = tf.one_hot(self.y, 10)
        logits = self.get_logits_cnn(x_exp, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
#         w = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
        y_prob = tf.nn.softmax(logits)
        y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
        return loss, y_hat