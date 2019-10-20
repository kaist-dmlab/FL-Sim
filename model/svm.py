import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

class Model(AbstractModel):
    
    def createModel(self, w_, x_shape_, x, y, callCnt=0):
        if w_ == None:
            x_shape = [ x_shape_[j] for j in range(1, len(x_shape_)) ] + [1]
            A = tf.get_variable('A' + str(callCnt), shape=x_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
            b = tf.get_variable('b' + str(callCnt), shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
        else:
            A = tf.get_variable('A' + str(callCnt), dtype=tf.float32, initializer=w_[0])
            b = tf.get_variable('b' + str(callCnt), dtype=tf.float32, initializer=w_[1])

        # formula (y = ax - b)
        formula = tf.matmul(x, A) - b

        # Loss = summation(max(0, 1 - pred*actual)) + alpha * L2_norm(A)^2
        y_reshaped = tf.reshape(y, [-1, 1]) # rank 증가시켜서 맞추기
        loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - formula * tf.cast(y_reshaped, tf.float32)))
        
        y_hat = tf.squeeze(tf.cast(tf.sign(formula), tf.int32))
        return [A, b], loss, y_hat
    
    def gradients(self, loss, g_, callCnt=0):
        gA = tf.get_variable('gA' + str(callCnt), dtype=tf.float32, initializer=g_[0])
        gb = tf.get_variable('gb' + str(callCnt), dtype=tf.float32, initializer=g_[1])
        return [gA, gb]