import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

class Model(AbstractModel):

    def get_logits_2nn(self, x, W_0, b_0, W_1, b_1, W_2, b_2):
        hidden_output_0 = tf.nn.relu(tf.matmul(x, W_0) + b_0)
        hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0, W_1) + b_1)
        logits = tf.matmul(hidden_output_1, W_2) + b_2
        return logits
    
    def createModel(self):
        if len(self.x_shape) != 2: raise Exception(self.x_shape)
        w_ = self.getInitVars()
#         W_0 = tf.get_variable('W_0', dtype=tf.float32, initializer=tf.random_normal([num_features, 200], stddev=(1/tf.sqrt(float(num_features)))))
#         b_0 = tf.get_variable('b_0', dtype=tf.float32, initializer=tf.random_normal([200]))
#         W_1 = tf.get_variable('W_1', dtype=tf.float32, initializer=tf.random_normal([200, 200], stddev=(1/tf.sqrt(float(200)))))
#         b_1 = tf.get_variable('b_1', dtype=tf.float32, initializer=tf.random_normal([200]))
#         W_2 = tf.get_variable('W_2', dtype=tf.float32, initializer=tf.random_normal([200, 10], stddev=(1/tf.sqrt(float(200)))))
#         b_2 = tf.get_variable('b_2', dtype=tf.float32, initializer=tf.random_normal([10]))
        W_0 = tf.get_variable('W_0', dtype=tf.float32, initializer=w_[0])
        b_0 = tf.get_variable('b_0', dtype=tf.float32, initializer=w_[1])
        W_1 = tf.get_variable('W_1', dtype=tf.float32, initializer=w_[2])
        b_1 = tf.get_variable('b_1', dtype=tf.float32, initializer=w_[3])
        W_2 = tf.get_variable('W_2', dtype=tf.float32, initializer=w_[4])
        b_2 = tf.get_variable('b_2', dtype=tf.float32, initializer=w_[5])
        y_onehot = tf.one_hot(self.y, 10)
        logits = self.get_logits_2nn(self.x, W_0, b_0, W_1, b_1, W_2, b_2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
        w = [W_0, b_0, W_1, b_1, W_2, b_2]
        y_prob = tf.nn.softmax(logits)
        y_hat = tf.squeeze(tf.cast(tf.argmax(y_prob, 1), tf.int32))
        return loss, w, y_hat
    
    def getInitVars(self):
        np.random.seed(1234) # 모든 노드가 같은 Initial Random Seed 를 갖지 않으면 학습되지 않음 (FedAvg 논문 참조)
        num_features = self.x_shape[1]
        W_0_ = np.random.normal(0, (1/np.sqrt(float(num_features))), num_features*200).reshape((num_features, 200)).astype(np.float32)
        b_0_ = np.random.normal(0, 1, 200).astype(np.float32)
        W_1_ = np.random.normal(0, (1/np.sqrt(float(200))), 200*200).reshape((200, 200)).astype(np.float32)
        b_1_ = np.random.normal(0, 1, 200).astype(np.float32)
        W_2_ = np.random.normal(0, (1/np.sqrt(float(200))), 200*10).reshape((200, 10)).astype(np.float32)
        b_2_ = np.random.normal(0, 1, 10).astype(np.float32)
        return [W_0_, b_0_, W_1_, b_1_, W_2_, b_2_]