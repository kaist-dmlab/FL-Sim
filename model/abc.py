import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod

from leaf.utils.tf_utils import graph_size

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

class AbstractModel(ABC):
    
    def __init__(self, args, x_shape):
        self.args = args
        self.x_shape = x_shape
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.graph = tf.Graph()
        with self.graph.as_default():
#             tf.set_random_seed(123 + seed)
            x_ph_shape = [None] + [ x_shape[j] for j in range(1, len(x_shape)) ]
            self.x = tf.placeholder(name='x', shape=x_ph_shape, dtype=tf.float32)
            self.y = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
            self.lr = tf.placeholder(name='lr', shape=[], dtype=tf.float32)
            self.loss, self.w, self.y_hat = self.createModel()
            self.g = tf.gradients(self.loss, self.w)
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_hat, self.y), tf.float32))
        self.sess = tf.Session(config=config, graph=self.graph)
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            
    def __del__(self):
        self.sess.close()
        
    def setParams(self, w_):
        with self.graph.as_default():
            allVars = tf.trainable_variables()
            for var, value in zip(allVars, w_):
                var.load(value, self.sess)
                
    def getParams(self):
        with self.graph.as_default():
            w_ = self.sess.run(tf.trainable_variables())
        return w_
    
    @abstractmethod
    def createModel(self):
        pass
    
    def local_train(self, dataBatch_i_, lr_, tau1_):
        with self.graph.as_default():
            w_byTime = []
            if not self.args.sgdEnabled:
                for _ in range(tau1_):
                    self.sess.run(self.train_op, feed_dict={self.lr: lr_, self.x: dataBatch_i_['x'], self.y: dataBatch_i_['y']})
                    w_ = self.sess.run(self.w, feed_dict={self.lr: lr_, self.x: dataBatch_i_['x'], self.y: dataBatch_i_['y']})
                    w_byTime.append(w_)
                #       loss_ = sess.run(loss, feed_dict={lr: lr_, x: dataBatch_i_['x'], y: dataBatch_i_['y']})
                #       print(lr_, loss_)
                    lr_ *= self.args.lrDecayRate
            else:
                numEpochSamples = dataBatch_i_['x'].shape[0]
                numItersPerEpoch = int(np.ceil(numEpochSamples / self.args.batchSize).tolist())
                for t1 in range(tau1_):
                    for it in range(numItersPerEpoch):
                        (sampleBatch_x, sampleBatch_y) = next_batch(self.args.batchSize, dataBatch_i_)
                        self.sess.run(self.train_op, feed_dict={self.lr: lr_, self.x: sampleBatch_x, self.y: sampleBatch_y})
                    # Epoch 마다 정보 저장
                    w_ = self.sess.run(self.w, feed_dict={self.lr: lr_, self.x: sampleBatch_x, self.y: sampleBatch_y})
                    w_byTime.append(w_)
                #       vs = sess.run(tf.concat([ tf.reshape(layer, [-1]) for layer in tf.get_collection(tf.GraphKeys.VARIABLES) ], axis=0))
                #       numVars = vs.shape ; print(numVars)
                    lr_ *= self.args.lrDecayRate
        return w_byTime, w_
    
    def federated_train_collect(self, w, trainData_byNid, lr, tau1):
        w_byTime_byNid = [] ; w_last_byNid = []
        for dataBatch_i in trainData_byNid:
            # 모든 노드의 모델을 주어진 모델 값으로 초기화
            self.setParams(w)
            
            # Train
            (w_byTime, w_last) = self.local_train(dataBatch_i, lr, tau1)
            w_byTime_byNid.append(w_byTime)
            w_last_byNid.append(w_last)
        return w_byTime_byNid, w_last_byNid
    
    def federated_train(self, w, trainData_byNid, lr, tau1, weight_byNid):
        (w_byTime_byNid, w_last_byNid) = self.federated_train_collect(w, trainData_byNid, lr, tau1)
        w_byTime = self.federated_aggregate(w_byTime_byNid, weight_byNid)
        return w_byTime, w_last_byNid
    
    def federated_aggregate(self, w_byTime_byNid, weight_byNid):
        w_byTime_byNid = np.array(w_byTime_byNid)
        w_byTime = [ average_w(w_byTime_byNid[:, col], weight_byNid) for col in range(len(w_byTime_byNid[0])) ]
        return w_byTime
    
    def evaluate(self, w_, dataBatch_):
        # 모든 노드의 모델을 주어진 모델 값으로 초기화
        self.setParams(w_)
        
        with self.graph.as_default():
            # Batch Size 만큼 나눠서 평가
            numTestIters = min(self.args.numTestIters, int(len(dataBatch_['x'])/self.args.batchSize))
            loss_ = [] ; accs_ = [] ; idxBegin = 0 ; idxEnd = 0
            for i in range(numTestIters):
                idxEnd += self.args.batchSize
                sampleBatch_x = dataBatch_['x'][idxBegin:idxEnd]
                sampleBatch_y = dataBatch_['y'][idxBegin:idxEnd]
                (loss_, acc_) = self.sess.run((self.loss, self.accuracy), feed_dict={self.x: sampleBatch_x, self.y: sampleBatch_y})
                accs_.append(acc_)
                idxBegin = idxEnd
        return np.mean(loss_), np.mean(accs_)
    
    def local_gradients(self, w_, dataBatch_):
        # 모든 노드의 모델을 주어진 모델 값으로 초기화
        self.setParams(w_)
        
        with self.graph.as_default():
            g_ = self.sess.run(self.g, feed_dict={self.x: dataBatch_['x'], self.y: dataBatch_['y']})
        return g_
    
    def federated_collect_gradients(self, w, nid2_Data_i):
        gs = [] ; nid2_gs = {}
        for nid in nid2_Data_i.keys():
            g = self.local_gradients(w, nid2_Data_i[nid])
            g = np_flatten(g)
            gs.append(g)
            nid2_gs[nid] = g
        return gs, nid2_gs