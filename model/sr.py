import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

class Model(AbstractModel):
    
    def __init__(self, args, trainData_by1Nid, testData_by1Nid):
        trainData_by1Nid[0]['x'] = np.array([ x_.flatten() for x_ in trainData_by1Nid[0]['x'] ], dtype=np.float32)
        testData_by1Nid[0]['x'] = np.array([ x_.flatten() for x_ in testData_by1Nid[0]['x'] ], dtype=np.float32)
        super().__init__(args, trainData_by1Nid, testData_by1Nid)
    
    def createModel(self):
        W_shape = [ self.x_shape[j] for j in range(1, len(self.x_shape)) ] + [10]
        W = tf.get_variable('W', shape=W_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        b = tf.get_variable('b', shape=[self.numClasses], dtype=tf.float32, initializer=tf.zeros_initializer())
        
        y_prob = tf.nn.softmax(tf.matmul(self.x, W) + b)
        loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(self.y, self.numClasses) * tf.math.log(y_prob), axis=[1]))
        y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
        return loss, y_hat