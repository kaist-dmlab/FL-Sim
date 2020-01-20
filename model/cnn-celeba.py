import tensorflow as tf
import numpy as np

from model.abc import AbstractModel

# https://github.com/TalwalkarLab/leaf/blob/master/models/celeba/cnn.py
class Model(AbstractModel):
    
    def createModel(self):
        out = self.x
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        logits = tf.layers.dense(out, self.numClasses)
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        y_hat = tf.cast(tf.argmax(logits, 1), tf.int32)
        return loss, y_hat