import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

class Model(AbstractModel):
    
    def createModel(self, w_, x_shape_, x, y, callCnt=0):
        if w_ == None:
            x_shape = [ x_shape_[j] for j in range(1, len(x_shape_)) ] + [10]
            W = tf.get_variable('W' + str(callCnt), shape=x_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
            b = tf.get_variable('b' + str(callCnt), shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer())
        else:
            W = tf.get_variable('W' + str(callCnt), dtype=tf.float32, initializer=w_[0])
            b = tf.get_variable('b' + str(callCnt), dtype=tf.float32, initializer=w_[1])
        y_prob = tf.nn.softmax(tf.matmul(x, W) + b)
        loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(y, 10) * tf.math.log(y_prob), axis=[1]))
        
        y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
        return [W, b], loss, y_hat
    
    def gradients(self, loss, g_, callCnt=0):
        gW = tf.get_variable('gW' + str(callCnt), dtype=tf.float32, initializer=g_[0])
        gb = tf.get_variable('gb' + str(callCnt), dtype=tf.float32, initializer=g_[1])
        return [gW, gb]