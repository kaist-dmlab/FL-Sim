import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

class Model(AbstractModel):

    def get_logits_2nn(self, x, W_0, b_0, W_1, b_1, W_2, b_2):
        hidden_output_0 = tf.nn.relu(tf.matmul(x, W_0) + b_0)
        hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0, W_1) + b_1)
        logits = tf.matmul(hidden_output_1, W_2) + b_2
        return logits
    
    def loss(self, w_, x_shape_, x, y, callCnt=0):
        if w_ == None:
            tf.random.set_random_seed(1234) # 모든 노드가 같은 Initial Random Seed 를 갖지 않으면 학습되지 않음 (FedAvg 논문 참조)
            if len(x_shape_) != 2: raise Exception(x_shape_)
            num_features = x_shape_[1]
            x_shape = [num_features, 200]
            W_0 = tf.get_variable('W_0' + str(callCnt), dtype=tf.float32, initializer=tf.random_normal(x_shape, stddev=(1/tf.sqrt(float(num_features)))))
            b_0 = tf.get_variable('b_0' + str(callCnt), dtype=tf.float32, initializer=tf.random_normal([200]))
            W_1 = tf.get_variable('W_1' + str(callCnt), dtype=tf.float32, initializer=tf.random_normal([200, 200], stddev=(1/tf.sqrt(float(200)))))
            b_1 = tf.get_variable('b_1' + str(callCnt), dtype=tf.float32, initializer=tf.random_normal([200]))
            W_2 = tf.get_variable('W_2' + str(callCnt), dtype=tf.float32, initializer=tf.random_normal([200, 10], stddev=(1/tf.sqrt(float(200)))))
            b_2 = tf.get_variable('b_2' + str(callCnt), dtype=tf.float32, initializer=tf.random_normal([10]))
        else:
            W_0 = tf.get_variable('W_0' + str(callCnt), dtype=tf.float32, initializer=w_[0])
            b_0 = tf.get_variable('b_0' + str(callCnt), dtype=tf.float32, initializer=w_[1])
            W_1 = tf.get_variable('W_1' + str(callCnt), dtype=tf.float32, initializer=w_[2])
            b_1 = tf.get_variable('b_1' + str(callCnt), dtype=tf.float32, initializer=w_[3])
            W_2 = tf.get_variable('W_2' + str(callCnt), dtype=tf.float32, initializer=w_[4])
            b_2 = tf.get_variable('b_2' + str(callCnt), dtype=tf.float32, initializer=w_[5])
        y_onehot = tf.one_hot(y, 10)
        logits = self.get_logits_2nn(x, W_0, b_0, W_1, b_1, W_2, b_2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
        return loss, [W_0, b_0, W_1, b_1, W_2, b_2]
    
    def predict(self, w, x):
        logits = self.get_logits_2nn(x, w[0], w[1], w[2], w[3], w[4], w[5])
        y_prob = tf.nn.softmax(logits)
        y_hat = tf.squeeze(tf.cast(tf.argmax(y_prob, 1), tf.int32))
        return y_hat
    
    def gradients(self, loss, g_, callCnt=0):
        gW_0 = tf.get_variable('W_0' + str(callCnt), dtype=tf.float32, initializer=g_[0])
        gb_0 = tf.get_variable('b_0' + str(callCnt), dtype=tf.float32, initializer=g_[1])
        gW_1 = tf.get_variable('W_1' + str(callCnt), dtype=tf.float32, initializer=g_[2])
        gb_1 = tf.get_variable('b_1' + str(callCnt), dtype=tf.float32, initializer=g_[3])
        gW_2 = tf.get_variable('W_2' + str(callCnt), dtype=tf.float32, initializer=g_[4])
        gb_2 = tf.get_variable('b_2' + str(callCnt), dtype=tf.float32, initializer=g_[5])
        return [gW_0, gb_0, gW_1, gb_1, gW_2, gb_2]