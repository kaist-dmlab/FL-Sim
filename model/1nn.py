import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

import fl_data

class Model(AbstractModel):
    
    def __init__(self, args, trainData_by1Nid, testData_by1Nid):
        fl_data.flattenX(trainData_by1Nid)
        fl_data.flattenX(testData_by1Nid)
        super().__init__(args, trainData_by1Nid, testData_by1Nid)
        
    def get_logits_2nn(self, x, W_0, b_0, W_1, b_1):
        hidden_output_0 = tf.nn.relu(tf.matmul(x, W_0) + b_0)
        logits = tf.matmul(hidden_output_0, W_1) + b_1
        return logits
    
    def createModel(self):
        if len(self.x_shape) != 2: raise Exception(self.x_shape)
        tf.set_random_seed(1234) # 모든 노드가 같은 Initial Random Seed 를 갖지 않으면 학습되지 않음 (FedAvg 논문 참조)
        num_features = self.x_shape[1]
        W_0 = tf.get_variable('W_0', dtype=tf.float32, initializer=tf.random_normal([num_features, 50], stddev=(1/tf.sqrt(float(num_features)))))
        b_0 = tf.get_variable('b_0', dtype=tf.float32, initializer=tf.random_normal([50]))
        W_1 = tf.get_variable('W_1', dtype=tf.float32, initializer=tf.random_normal([50, 10], stddev=(1/tf.sqrt(float(50)))))
        b_1 = tf.get_variable('b_1', dtype=tf.float32, initializer=tf.random_normal([10]))
        
        y_onehot = tf.one_hot(self.y, 10)
        logits = self.get_logits_2nn(self.x, W_0, b_0, W_1, b_1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
#         w = [W_0, b_0, W_1, b_1, W_2, b_2]
        y_prob = tf.nn.softmax(logits)
        y_hat = tf.squeeze(tf.cast(tf.argmax(y_prob, 1), tf.int32))
        return loss, y_hat