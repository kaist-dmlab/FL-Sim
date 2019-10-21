import tensorflow as tf
import numpy as np

from model.cnn import Model as CnnModel

class Model(CnnModel):
    
    def createModel(self):
        tf.set_random_seed(1234) # 모든 노드가 같은 Initial Random Seed 를 갖지 않으면 학습되지 않음 (FedAvg 논문 참조)
        W_conv1 = tf.get_variable('W_conv1', dtype=tf.float32, initializer=tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
        b_conv1 = tf.get_variable('b_conv1', dtype=tf.float32, initializer=tf.constant(0.0, shape=[64]))
        W_conv2 = tf.get_variable('W_conv2', dtype=tf.float32, initializer=tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
        b_conv2 = tf.get_variable('b_conv2', dtype=tf.float32, initializer=tf.constant(0.0, shape=[64]))
        W_fc1 = tf.get_variable('W_fc1', dtype=tf.float32, initializer=tf.truncated_normal(shape=[1600, 256], stddev=5e-2))
        b_fc1 = tf.get_variable('b_fc1', dtype=tf.float32, initializer=tf.constant(0.0, shape=[256]))
        W_fc2 = tf.get_variable('W_fc2', dtype=tf.float32, initializer=tf.truncated_normal(shape=[256, 10], stddev=5e-2))
        b_fc2 = tf.get_variable('b_fc2', dtype=tf.float32, initializer=tf.constant(0.0, shape=[10]))
        
        y_onehot = tf.one_hot(self.y, 10)
        logits = self.get_logits_cnn(self.x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
        w = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
        y_prob = tf.nn.softmax(logits)
        y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
        return loss, w, y_hat