import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod

from leaf.models.utils.tf_utils import graph_size

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
    
    def __init__(self, args, trainData_by1Nid, testData_by1Nid):
        self.args = args
        self.trainData_by1Nid = trainData_by1Nid
        self.testData_by1Nid = testData_by1Nid
        self.x_shape = trainData_by1Nid[0]['x'].shape
        self.numClasses = len(np.unique(trainData_by1Nid[0]['y']))
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + args.seed)
            x_ph_shape = [None] + [ self.x_shape[j] for j in range(1, len(self.x_shape)) ]
            self.x = tf.placeholder(name='x', shape=x_ph_shape, dtype=tf.float32)
            self.y = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
            self.lr = tf.placeholder(name='lr', dtype=tf.float32)
            self.loss, self.y_hat = self.createModel()
            self.w = tf.trainable_variables()
            self.g = tf.gradients(self.loss, self.w)
            self.train_op = self.getOptimizer()
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_hat), tf.float32))
#             self.accuracy = tf.reduce_mean(tf.count_nonzero(tf.equal(self.y, self.y_hat)))
        self.sess = tf.Session(config=config, graph=self.graph)
        
        self.size = graph_size(self.graph)
        
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flopOps = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
            
    def __del__(self):
        self.sess.close()
        
    def setParams(self, w):
        with self.graph.as_default():
            allVars = tf.trainable_variables()
            for var, value in zip(allVars, w):
                var.load(value, self.sess)
                
    def getParams(self):
        with self.graph.as_default():
            w = self.sess.run(tf.trainable_variables())
        return w
    
    @abstractmethod
    def createModel(self):
        pass
    
    def getOptimizer(self):
        return tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
    
    def augment(self, x):
        return x
    
    def local_train(self, dataBatch, lr, tau1):
        with self.graph.as_default():
            w_byTime = []
            if not self.args.sgdEnabled:
                for _ in range(tau1):
                    dataBatch_x = dataBatch['x']
                    dataBatch_y = dataBatch['y']
                    dataBatch_x = self.augment(dataBatch_x)
                    self.sess.run(self.train_op, feed_dict={self.lr: lr, self.x: dataBatch_x, self.y: dataBatch_y})
                    w = self.sess.run(self.w, feed_dict={self.lr: lr, self.x: dataBatch_x, self.y: dataBatch_y})
                    w_byTime.append(w)
                    lr *= self.args.lrDecayRate
            else:
                numEpochSamples = dataBatch['x'].shape[0]
                numItersPerEpoch = int(np.ceil(numEpochSamples / self.args.batchSize).tolist())
                for t1 in range(tau1):
                    for it in range(numItersPerEpoch):
                        (sampleBatch_x, sampleBatch_y) = next_batch(self.args.batchSize, dataBatch)
                        sampleBatch_x = self.augment(sampleBatch_x)
                        self.sess.run(self.train_op, feed_dict={self.lr: lr, self.x: sampleBatch_x, self.y: sampleBatch_y})
                    # Epoch 마다 정보 저장
                    w = self.sess.run(self.w, feed_dict={self.lr: lr, self.x: sampleBatch_x, self.y: sampleBatch_y})
                    w_byTime.append(w)
                    lr *= self.args.lrDecayRate
        return w_byTime, w
    
    def federated_train_collect(self, w, trainData_byNid, lr, tau1):
        w_byTime_byNid = [] ; w_last_byNid = []
        for dataBatch in trainData_byNid:
            # 모든 노드의 모델을 주어진 모델 값으로 초기화
            self.setParams(w)
            
            # Train
            (w_byTime, w_last) = self.local_train(dataBatch, lr, tau1)
            w_byTime_byNid.append(w_byTime)
            w_last_byNid.append(w_last)
        # https://github.com/TalwalkarLab/leaf/blob/fb5b1da258a4d332fe8dbc15c656dad82ed8a20d/models/model.py#L91
#         comp = tau1 * len(trainData_byNid[0]['y']) * self.flops
        return w_byTime_byNid, w_last_byNid
    
    def federated_train(self, w, trainData_byNid, lr, tau1, weight_byNid):
        (w_byTime_byNid, w_last_byNid) = self.federated_train_collect(w, trainData_byNid, lr, tau1)
        w_byTime = self.federated_aggregate(w_byTime_byNid, weight_byNid)
        return w_byTime, w_last_byNid
    
    def federated_aggregate(self, w_byTime_byNid, weight_byNid):
        w_byTime_byNid = np.array(w_byTime_byNid)
        w_byTime = [ average_w(w_byTime_byNid[:, col], weight_byNid) for col in range(len(w_byTime_byNid[0])) ]
        return w_byTime
    
    def evaluate_batch(self, w, dataBatch, numSamples):
        # 모든 노드의 모델을 주어진 모델 값으로 초기화
        self.setParams(w)
        
        with self.graph.as_default():
            # Batch Size 만큼 나눠서 평가
            minSamples = min(numSamples, len(dataBatch['x']))
            numIters = int(minSamples / self.args.batchSize)
            losses_ = [] ; accs_ = [] ; idxBegin = 0 ; idxEnd = 0
            for i in range(numIters):
                idxEnd += self.args.batchSize
                sampleBatch_x = dataBatch['x'][idxBegin:idxEnd]
                sampleBatch_y = dataBatch['y'][idxBegin:idxEnd]
                loss_, acc_ = self.sess.run((self.loss, self.accuracy), feed_dict={self.x: sampleBatch_x, self.y: sampleBatch_y})
                losses_.append(loss_)
                accs_.append(acc_)
                idxBegin = idxEnd
        return np.mean(losses_), np.mean(accs_)
    
    def evaluate(self, w):
        loss_train, acc_train = self.evaluate_batch(w, self.trainData_by1Nid[0], len(self.trainData_by1Nid[0]['x']))
        loss_test, acc_test = self.evaluate_batch(w, self.testData_by1Nid[0], len(self.testData_by1Nid[0]['x']))
        return loss_train, acc_train, loss_test, acc_test
    
    def local_gradients(self, w, dataBatch):
        # 모든 노드의 모델을 주어진 모델 값으로 초기화
        self.setParams(w)
        
        with self.graph.as_default():
            g_ = self.sess.run(self.g, feed_dict={self.x: dataBatch['x'], self.y: dataBatch['y']})
        return g_
    
    def federated_collect_gradients(self, w, nid2_Data_i):
        gs = [] ; nid2_gs = {}
        for nid in nid2_Data_i.keys():
            g = self.local_gradients(w, nid2_Data_i[nid])
            g = np_flatten(g)
            gs.append(g)
            nid2_gs[nid] = g
        return gs, nid2_gs