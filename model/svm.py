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
        A_shape = [ self.x_shape[j] for j in range(1, len(self.x_shape)) ] + [1]
        A = tf.get_variable('A', shape=A_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        b = tf.get_variable('b', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
        
        # formula (y = ax - b)
        formula = tf.matmul(self.x, A) - b
        
        # Loss = summation(max(0, 1 - pred*actual)) + alpha * L2_norm(A)^2
        y_reshaped = tf.reshape(self.y, [-1, 1]) # rank 증가시켜서 맞추기
        loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - formula * tf.cast(y_reshaped, tf.float32)))
#         w = [A, b]
        y_hat = tf.squeeze(tf.cast(tf.sign(formula), tf.int32))
        return loss, y_hat