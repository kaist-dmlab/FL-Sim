import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod

def next_batch(batchSize, dataBatch):
    x = dataBatch['x']
    y = dataBatch['y']
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x_randBatch = [x[i] for i in idx[:batchSize]]
    y_randBatch = [y[i] for i in idx[:batchSize]]
    return x_randBatch, y_randBatch

def np_flatten(model):
    model_ = []
    for i in range(len(model)):
        model_ += model[i].flatten().tolist()
    return np.array(model_)

def average_w(w_byNid, weight_byNid):
    weight_byNid = np.array(weight_byNid) / sum(weight_byNid)
    w_avg = []
    for col in range(w_byNid.shape[1]):
        sum_w = 0
        for row in range(w_byNid.shape[0]):
            sum_w += w_byNid[row, col] * weight_byNid[row]
        w_avg.append(sum_w)
    return w_avg

def federated_aggregate(w_byTime_byNid, weight_byNid):
    w_byTime_byNid = np.array(w_byTime_byNid)
    w_byTime = [ average_w(w_byTime_byNid[:,t], weight_byNid) for t in range(len(w_byTime_byNid[0])) ]
    return w_byTime

class AbstractModel(ABC):
    
    def __init__(self, args):
        self.args = args
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
    def createModel(self, w_, x_shape_, x, y, callCnt=0):
        pass
    
    def gradients(self, loss, g_, callCnt=0):
        pass
    
    def local_train(self, w_, dataBatch_i_, lr_, tau1_):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(config=self.config, graph=graph) as sess:
                x_shape_ = dataBatch_i_['x'].shape
                x_shape = [None] + [ x_shape_[j] for j in range(1, len(x_shape_)) ]
                x = tf.placeholder(name='x', shape=x_shape, dtype=tf.float32)
                y = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
                (w, loss, _) = self.createModel(w_, x_shape_, x, y)
                
                # 직접 gradients 계산하는 대신 Optimizer 에서 계산된 gradients 사용
                lr = tf.placeholder(name='lr', shape=[], dtype=tf.float32)
                train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
                
                sess.run(tf.global_variables_initializer())
                w_byTime = []
                if not self.args.sgdEnabled:
                    for _ in range(tau1_):
                        sess.run(train_op, feed_dict={lr: lr_, x: dataBatch_i_['x'], y: dataBatch_i_['y']})
                        w_ = sess.run(w, feed_dict={lr: lr_, x: dataBatch_i_['x'], y: dataBatch_i_['y']})
                        w_byTime.append(w_)
            #             loss_ = sess.run(loss, feed_dict={lr: lr_, x: dataBatch_i_['x'], y: dataBatch_i_['y']})
            #             print(lr_, loss_)
                        lr_ *= self.args.lrDecayRate
                else:
                    numEpochSamples = dataBatch_i_['x'].shape[0]
                    numItersPerEpoch = int(np.ceil(numEpochSamples / self.args.batchSize).tolist())
                    for t1 in range(tau1_):
                        for it in range(numItersPerEpoch):
                            (sampleBatch_x, sampleBatch_y) = next_batch(self.args.batchSize, dataBatch_i_)
                            sess.run(train_op, feed_dict={lr: lr_, x: sampleBatch_x, y: sampleBatch_y})
                        # Epoch 마다 정보 저장
                        w_ = sess.run(w, feed_dict={lr: lr_, x: sampleBatch_x, y: sampleBatch_y})
                        w_byTime.append(w_)
            #             vs = sess.run(tf.concat([ tf.reshape(layer, [-1]) for layer in tf.get_collection(tf.GraphKeys.VARIABLES) ], axis=0))
            #             numVars = vs.shape ; print(numVars)
                        lr_ *= self.args.lrDecayRate
        return w_byTime, w_
    
    def federated_train_collect(self, w, trainData_byNid, lr, tau1):
        w_byTime_byNid = [] ; w_last_byNid = []
        for dataBatch_i in trainData_byNid:
            (w_byTime, w_last) = self.local_train(w, dataBatch_i, lr, tau1)
            w_byTime_byNid.append(w_byTime)
            w_last_byNid.append(w_last)
        return w_byTime_byNid, w_last_byNid
    
    def federated_train(self, w, trainData_byNid, lr, tau1, weight_byNid):
        (w_byTime_byNid, w_last_byNid) = self.federated_train_collect(w, trainData_byNid, lr, tau1)
        w_byTime = federated_aggregate(w_byTime_byNid, weight_byNid)
        return w_byTime, w_last_byNid
    
    def evaluate(self, w_, dataBatch_):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(config=self.config, graph=graph) as sess:
                x_shape_ = dataBatch_['x'].shape
                x_shape = [None] + [ x_shape_[j] for j in range(1, len(x_shape_)) ]
                x = tf.placeholder(name='x', shape=x_shape, dtype=tf.float32)
                y = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
                (w, loss, y_hat) = self.createModel(w_, x_shape_, x, y)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y_hat, y), tf.float32))
                
                sess.run(tf.global_variables_initializer())
                
                # Batch Size 만큼 나눠서 평가
                numTestIters = min(self.args.numTestIters, int(len(dataBatch_['x'])/self.args.batchSize))
                loss_ = [] ; accs_ = [] ; idxBegin = 0 ; idxEnd = 0
                for i in range(numTestIters):
                    idxEnd += self.args.batchSize
                    sampleBatch_x = dataBatch_['x'][idxBegin:idxEnd]
                    sampleBatch_y = dataBatch_['y'][idxBegin:idxEnd]
                    (loss_, acc_) = sess.run((loss, accuracy), feed_dict={x: sampleBatch_x, y: sampleBatch_y})
                    accs_.append(acc_)
                    idxBegin = idxEnd
        return np.mean(loss_), np.mean(accs_)
    
    def local_gradients(self, w_, dataBatch_):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(config=self.config, graph=graph) as sess:
                x_shape_ = dataBatch_['x'].shape
                x_shape = [None] + [ x_shape_[j] for j in range(1, len(x_shape_)) ]
                x = tf.placeholder(name='x', shape=x_shape, dtype=tf.float32)
                y = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
                (w, loss, _) = self.createModel(w_, x_shape_, x, y)
                g = tf.gradients(loss, w)
                
                sess.run(tf.global_variables_initializer())
                g_ = sess.run(g, feed_dict={x: dataBatch_['x'], y: dataBatch_['y']})
        return g_
    
    def federated_collect_gradients(self, w, nid2_Data_i):
        gs = [] ; nid2_gs = {}
        for nid in nid2_Data_i.keys():
            g = self.local_gradients(w, nid2_Data_i[nid])
            g = np_flatten(g)
            gs.append(g)
            nid2_gs[nid] = g
        return gs, nid2_gs