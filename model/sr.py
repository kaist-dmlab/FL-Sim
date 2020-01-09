import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

import fl_data

class Model(AbstractModel):
    
    def __init__(self, args, trainData_by1Nid, testData_by1Nid):
        fl_data.flattenX(trainData_by1Nid)
        fl_data.flattenX(testData_by1Nid)
        super().__init__(args, trainData_by1Nid, testData_by1Nid)
    
    def createModel(self):
        W_shape = [ self.x_shape[j] for j in range(1, len(self.x_shape)) ] + [10]
        W = tf.get_variable('W', shape=W_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        b = tf.get_variable('b', shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer())
        
        y_prob = tf.nn.softmax(tf.matmul(self.x, W) + b)
        loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(self.y, 10) * tf.math.log(y_prob), axis=[1]))
#         w = [W, b]
        y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
        return loss, y_hat