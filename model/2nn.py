import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

class Model(AbstractModel):
    
    def __init__(self, args, trainData_by1Nid, testData_by1Nid):
        trainData_by1Nid[0]['x'] = np.array([ x_.flatten() for x_ in trainData_by1Nid[0]['x'] ], dtype=np.float32)
        testData_by1Nid[0]['x'] = np.array([ x_.flatten() for x_ in testData_by1Nid[0]['x'] ], dtype=np.float32)
        super().__init__(args, trainData_by1Nid, testData_by1Nid)

    def get_logits_2nn(self, x, W_0, b_0, W_1, b_1, W_2, b_2):
        hidden_output_0 = tf.nn.relu(tf.matmul(x, W_0) + b_0)
        hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0, W_1) + b_1)
        logits = tf.matmul(hidden_output_1, W_2) + b_2
        return logits
    
    def createModel(self):
        if len(self.x_shape) != 2: raise Exception(self.x_shape)
        num_features = self.x_shape[1]
        W_0 = tf.get_variable('W_0', dtype=tf.float32, initializer=tf.random_normal([num_features, 200], stddev=(1/tf.sqrt(float(num_features)))))
        b_0 = tf.get_variable('b_0', dtype=tf.float32, initializer=tf.random_normal([200]))
        W_1 = tf.get_variable('W_1', dtype=tf.float32, initializer=tf.random_normal([200, 200], stddev=(1/tf.sqrt(float(200)))))
        b_1 = tf.get_variable('b_1', dtype=tf.float32, initializer=tf.random_normal([200]))
        W_2 = tf.get_variable('W_2', dtype=tf.float32, initializer=tf.random_normal([200, self.numClasses], stddev=(1/tf.sqrt(float(200)))))
        b_2 = tf.get_variable('b_2', dtype=tf.float32, initializer=tf.random_normal([self.numClasses]))
        
        y_onehot = tf.one_hot(self.y, self.numClasses)
        logits = self.get_logits_2nn(self.x, W_0, b_0, W_1, b_1, W_2, b_2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
        y_prob = tf.nn.softmax(logits)
        y_hat = tf.squeeze(tf.cast(tf.argmax(y_prob, 1), tf.int32))
        return loss, y_hat