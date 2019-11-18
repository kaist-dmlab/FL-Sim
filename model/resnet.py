import tensorflow as tf

import numpy as np
import random

from model.abc import AbstractModel

def data_augmentation(batch, img_size):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [img_size, img_size], 4)
    return batch

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

weight_init = tf.contrib.layers.variance_scaling_initializer()
weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)
        return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x

def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = tf.nn.relu(x)
        
        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            
        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = tf.nn.relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')
        return x + x_init

def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = tf.nn.relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = tf.nn.relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = tf.nn.relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')
        return x + shortcut
    
def get_residual_layer(resN) :
    x = []
    if resN == 18 :
        x = [2, 2, 2, 2]
    elif resN == 34 :
        x = [3, 4, 6, 3]
    elif resN == 50 :
        x = [3, 4, 6, 3]
    elif resN == 101 :
        x = [3, 4, 23, 3]
    elif resN == 152 :
        x = [3, 8, 36, 3]
    else:
        raise Exception(resN)
    return x

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)
        
def network(resN, x, is_training=True, reuse=False):
    with tf.variable_scope("network", reuse=reuse):
        if resN < 50 :
            residual_block = resblock
        else :
            residual_block = bottle_resblock
        residual_list = get_residual_layer(resN)
        
        ch = 32 # paper is 64
        x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')
        for i in range(residual_list[0]) :
            x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))
            
        x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')
        for i in range(1, residual_list[1]) :
            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))
            
        x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')
        for i in range(1, residual_list[2]) :
            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))
            
        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')
        for i in range(1, residual_list[3]) :
            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))
            
        x = batch_norm(x, is_training, scope='batch_norm')
        x = tf.nn.relu(x)
        
        x = global_avg_pooling(x)
        x = fully_conneted(x, units=10, scope='logit')
        return x
        
def classification_loss(y, logits):
    y_onehot = tf.one_hot(y, 10)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
    y_prob = tf.nn.softmax(logits)
    y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
    return loss, y_hat

class ResNet(object):
    def __init__(self, resN):
        self.resN = resN
        
    def build_model(self):
        """ Graph Input """
        self.train_x = tf.placeholder(name='train_x', shape=[None, 32, 32, 3], dtype=tf.float32)
        self.train_y = tf.placeholder(name='train_y', shape=[None], dtype=tf.int32)
        
        self.test_x = tf.placeholder(name='test_x', shape=[None, 32, 32, 3], dtype=tf.float32)
        self.test_y = tf.placeholder(name='test_y', shape=[None], dtype=tf.int32)
        
        self.lr = tf.placeholder(name='lr', dtype=tf.float32)
        
        train_logits = network(self.resN, self.train_x)
        test_logits = network(self.resN, self.test_x, is_training=False, reuse=True)
        
        train_loss, _ = classification_loss(y=self.train_y, logits=train_logits)
        _, test_y_hat = classification_loss(y=self.test_y, logits=test_logits)
        return train_loss, test_y_hat

class Model(AbstractModel):

    def createModel(self):
        resN = 18
#         self.test_x = tf.placeholder(name='test_x', shape=[None, 32, 32, 3], dtype=tf.float32)
#         self.test_y = tf.placeholder(name='test_y', shape=[None], dtype=tf.int32)
        
        train_logits = network(resN, self.x)
#         test_logits = network(resN, self.test_x, is_training=False, reuse=True)
        
        train_loss, train_y_hat = classification_loss(y=self.y, logits=train_logits)
#         _, test_y_hat = classification_loss(y=self.test_y, logits=test_logits)
        return train_loss, train_y_hat
    
    def getOptimizer(self):
        return tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.loss)
    
    def augment(self, x):
        return data_augmentation(x, 32)
    
#     def evaluate_batch(self, w, dataBatch, numSamples):
#         # 모든 노드의 모델을 주어진 모델 값으로 초기화
#         self.setParams(w)
        
#         with self.graph.as_default():
#             # Batch Size 만큼 나눠서 평가
#             minSamples = min(numSamples, len(dataBatch['x']))
#             numIters = int(minSamples / self.args.batchSize)
#             losses_ = [] ; accs_ = [] ; idxBegin = 0 ; idxEnd = 0
#             for i in range(numIters):
#                 idxEnd += self.args.batchSize
#                 sampleBatch_x = dataBatch['x'][idxBegin:idxEnd]
#                 sampleBatch_y = dataBatch['y'][idxBegin:idxEnd]
#                 loss_, acc_ = self.sess.run((self.loss, self.accuracy), feed_dict={self.x: sampleBatch_x, self.y: sampleBatch_y, \
#                                                                                    self.test_x: sampleBatch_x, self.test_y: sampleBatch_y})
#                 losses_.append(loss_)
#                 accs_.append(acc_)
#                 idxBegin = idxEnd
#         return np.mean(losses_), np.mean(accs_)