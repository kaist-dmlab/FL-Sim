import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

class Model(AbstractModel):
    
    def createModel(self):
        w_ = self.getInitVars();
#         W = tf.get_variable('W', shape=W_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
#         b = tf.get_variable('b', shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer())
        W = tf.get_variable('W', dtype=tf.float32, initializer=w_[0])
        b = tf.get_variable('b', dtype=tf.float32, initializer=w_[1])
        
        y_prob = tf.nn.softmax(tf.matmul(self.x, W) + b)
        loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(self.y, 10) * tf.math.log(y_prob), axis=[1]))
        w = [W, b]
        y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
        return loss, w, y_hat
    
    def getInitVars(self):
        W_shape = [ self.x_shape[j] for j in range(1, len(self.x_shape)) ] + [10]
        W_ = np.zeros(W_shape, dtype=np.float32)
        b_ = np.zeros(10, dtype=np.float32)
        return [W_, b_]