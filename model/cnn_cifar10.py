import tensorflow as tf
import numpy as np

from model.cnn import Model as CnnModel

class Model(CnnModel):
    
    def createModel(self):
        w_ = self.getInitVars()
        W_conv1 = tf.get_variable('W_conv1', dtype=tf.float32, initializer=w_[0])
        b_conv1 = tf.get_variable('b_conv1', dtype=tf.float32, initializer=w_[1])
        W_conv2 = tf.get_variable('W_conv2', dtype=tf.float32, initializer=w_[2])
        b_conv2 = tf.get_variable('b_conv2', dtype=tf.float32, initializer=w_[3])
        W_fc1 = tf.get_variable('W_fc1', dtype=tf.float32, initializer=w_[4])
        b_fc1 = tf.get_variable('b_fc1', dtype=tf.float32, initializer=w_[5])
        W_fc2 = tf.get_variable('W_fc2', dtype=tf.float32, initializer=w_[6])
        b_fc2 = tf.get_variable('b_fc2', dtype=tf.float32, initializer=w_[7])
        y_onehot = tf.one_hot(self.y, 10)
        logits = self.get_logits_cnn(self.x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
        w = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
        y_prob = tf.nn.softmax(logits)
        y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
        return loss, w, y_hat
    
    def getInitVars(self):
        np.random.seed(1234) # 모든 노드가 같은 Initial Random Seed 를 갖지 않으면 학습되지 않음 (FedAvg 논문 참조)
        num_features = self.x_shape[1]
        W_conv1_ = np.random.normal(0, 5e-2, 5*5*3*64).reshape((5, 5, 3, 64)).astype(np.float32)
        b_conv1_ = np.zeros(64, dtype=np.float32)
        W_conv2_ = np.random.normal(0, 5e-2, 5*5*64*64).reshape((5, 5, 64, 64)).astype(np.float32)
        b_conv2_ = np.zeros(64, dtype=np.float32)
        W_fc1_ = np.random.normal(0, 5e-2, 1600*256).reshape((1600, 256)).astype(np.float32)
        b_fc1_ = np.zeros(256, dtype=np.float32)
        W_fc2_ = np.random.normal(0, 5e-2, 256*10).reshape((256, 10)).astype(np.float32)
        b_fc2_ = np.zeros(10, dtype=np.float32)
        return [W_conv1_, b_conv1_, W_conv2_, b_conv2_, W_fc1_, b_fc1_, W_fc2_, b_fc2_]