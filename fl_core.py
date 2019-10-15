import tensorflow as tf
import numpy as np

import fl_param

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def loss_sr(w_, x_shape_, x, y, callCnt=0):
    if w_ == None:
        x_shape = [ x_shape_[j] for j in range(1, len(x_shape_)) ] + [10]
        W = tf.get_variable('W' + str(callCnt), shape=x_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        b = tf.get_variable('b' + str(callCnt), shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer())
    else:
        W = tf.get_variable('W' + str(callCnt), dtype=tf.float32, initializer=w_[0])
        b = tf.get_variable('b' + str(callCnt), dtype=tf.float32, initializer=w_[1])
    y_prob = tf.nn.softmax(tf.matmul(x, W) + b)
    loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(y, 10) * tf.math.log(y_prob), axis=[1]))
    return loss, [W, b]

def predict_sr(w, x):
    W = w[0]
    b = w[1]
    y_prob = tf.nn.softmax(tf.matmul(x, W) + b)
    y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
    return y_hat

def get_gradients_sr(loss, g_, callCnt=0):
    gW = tf.get_variable('gW' + str(callCnt), dtype=tf.float32, initializer=g_[0])
    gb = tf.get_variable('gb' + str(callCnt), dtype=tf.float32, initializer=g_[1])
    return [gW, gb]

def get_logits_cnn(x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc3, b_fc3):
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)
        
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    h_conv2 = tf.nn.relu(tf.nn.conv2d(norm1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
    norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    h_pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    h_pool2_flatten = tf.keras.layers.Flatten()(h_pool2)
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, W_fc1) + b_fc1)
    logits = tf.matmul(h_fc1,W_fc3) + b_fc3
    return logits

def loss_cnn(w_, x_shape_, x, y, callCnt=0):
    x = tf.expand_dims(x, axis=-1)
    if w_ == None:
        tf.random.set_random_seed(1234) # 모든 노드가 같은 Initial Random Seed 를 갖지 않으면 학습되지 않음 (FedAvg 논문 참조)
        W_conv1 = tf.get_variable('W_conv1' + str(callCnt), dtype=tf.float32, initializer=tf.truncated_normal(shape=[5, 5, 1, 64], stddev=5e-2))
        b_conv1 = tf.get_variable('b_conv1' + str(callCnt), dtype=tf.float32, initializer=tf.constant(0.0, shape=[64]))
        
        W_conv2 = tf.get_variable('W_conv2' + str(callCnt), dtype=tf.float32, initializer=tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
        b_conv2 = tf.get_variable('b_conv2' + str(callCnt), dtype=tf.float32, initializer=tf.constant(0.0, shape=[64]))
        
        W_fc1 = tf.get_variable('W_fc1' + str(callCnt), dtype=tf.float32, initializer=tf.truncated_normal(shape=[1024, 256], stddev=5e-2))
        b_fc1 = tf.get_variable('b_fc1' + str(callCnt), dtype=tf.float32, initializer=tf.constant(0.0, shape=[256]))
        
        W_fc3 = tf.get_variable('W_fc3' + str(callCnt), dtype=tf.float32, initializer=tf.truncated_normal(shape=[256, 10], stddev=5e-2))
        b_fc3 = tf.get_variable('b_fc3' + str(callCnt), dtype=tf.float32, initializer=tf.constant(0.0, shape=[10]))
    else:
        W_conv1 = tf.get_variable('W_conv1' + str(callCnt), dtype=tf.float32, initializer=w_[0])
        b_conv1 = tf.get_variable('b_conv1' + str(callCnt), dtype=tf.float32, initializer=w_[1])
        
        W_conv2 = tf.get_variable('W_conv2' + str(callCnt), dtype=tf.float32, initializer=w_[2])
        b_conv2 = tf.get_variable('b_conv2' + str(callCnt), dtype=tf.float32, initializer=w_[3])
        
        W_fc1 = tf.get_variable('W_fc1' + str(callCnt), dtype=tf.float32, initializer=w_[4])
        b_fc1 = tf.get_variable('b_fc1' + str(callCnt), dtype=tf.float32, initializer=w_[5])
        
        W_fc3 = tf.get_variable('W_fc3' + str(callCnt), dtype=tf.float32, initializer=w_[6])
        b_fc3 = tf.get_variable('b_fc3' + str(callCnt), dtype=tf.float32, initializer=w_[7])
    y_onehot = tf.one_hot(y, 10)
    logits = get_logits_cnn(x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc3, b_fc3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
    return loss, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc3, b_fc3]

def predict_cnn(w, x):
    x = tf.expand_dims(x, axis=-1)
    logits = get_logits_cnn(x, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
    y_prob = tf.nn.softmax(logits)
    y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
    return y_hat

def loss_cnn_cifar10(w_, x_shape_, x, y, callCnt=0):
    if w_ == None:
        tf.random.set_random_seed(1234) # 모든 노드가 같은 Initial Random Seed 를 갖지 않으면 학습되지 않음 (FedAvg 논문 참조)
        W_conv1 = tf.get_variable('W_conv1' + str(callCnt), dtype=tf.float32, initializer=tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
        b_conv1 = tf.get_variable('b_conv1' + str(callCnt), dtype=tf.float32, initializer=tf.constant(0.0, shape=[64]))
        
        W_conv2 = tf.get_variable('W_conv2' + str(callCnt), dtype=tf.float32, initializer=tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
        b_conv2 = tf.get_variable('b_conv2' + str(callCnt), dtype=tf.float32, initializer=tf.constant(0.0, shape=[64]))
        
        W_fc1 = tf.get_variable('W_fc1' + str(callCnt), dtype=tf.float32, initializer=tf.truncated_normal(shape=[1600, 256], stddev=5e-2))
        b_fc1 = tf.get_variable('b_fc1' + str(callCnt), dtype=tf.float32, initializer=tf.constant(0.0, shape=[256]))
        
        W_fc3 = tf.get_variable('W_fc3' + str(callCnt), dtype=tf.float32, initializer=tf.truncated_normal(shape=[256, 10], stddev=5e-2))
        b_fc3 = tf.get_variable('b_fc3' + str(callCnt), dtype=tf.float32, initializer=tf.constant(0.0, shape=[10]))
    else:
        W_conv1 = tf.get_variable('W_conv1' + str(callCnt), dtype=tf.float32, initializer=w_[0])
        b_conv1 = tf.get_variable('b_conv1' + str(callCnt), dtype=tf.float32, initializer=w_[1])
        
        W_conv2 = tf.get_variable('W_conv2' + str(callCnt), dtype=tf.float32, initializer=w_[2])
        b_conv2 = tf.get_variable('b_conv2' + str(callCnt), dtype=tf.float32, initializer=w_[3])
        
        W_fc1 = tf.get_variable('W_fc1' + str(callCnt), dtype=tf.float32, initializer=w_[4])
        b_fc1 = tf.get_variable('b_fc1' + str(callCnt), dtype=tf.float32, initializer=w_[5])
        
        W_fc3 = tf.get_variable('W_fc3' + str(callCnt), dtype=tf.float32, initializer=w_[6])
        b_fc3 = tf.get_variable('b_fc3' + str(callCnt), dtype=tf.float32, initializer=w_[7])
    y_onehot = tf.one_hot(y, 10)
    logits = get_logits_cnn(x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc3, b_fc3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
    return loss, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc3, b_fc3]

def predict_cnn_cifar10(w, x):
    logits = get_logits_cnn(x, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
    y_prob = tf.nn.softmax(logits)
    y_hat = tf.cast(tf.argmax(y_prob, 1), tf.int32)
    return y_hat

def get_gradients_cnn(loss, g_, callCnt=0):
    gW_conv1 = tf.get_variable('gW_conv1' + str(callCnt), dtype=tf.float32, initializer=g_[0])
    gb_conv1 = tf.get_variable('gb_conv1' + str(callCnt), dtype=tf.float32, initializer=g_[1])
    
    gW_conv2 = tf.get_variable('gW_conv2' + str(callCnt), dtype=tf.float32, initializer=g_[2])
    gb_conv2 = tf.get_variable('gb_conv2' + str(callCnt), dtype=tf.float32, initializer=g_[3])
    
    gW_fc1 = tf.get_variable('gW_fc1' + str(callCnt), dtype=tf.float32, initializer=g_[4])
    gb_fc1 = tf.get_variable('gb_fc1' + str(callCnt), dtype=tf.float32, initializer=g_[5])
    
    gW_fc3 = tf.get_variable('gW_fc3' + str(callCnt), dtype=tf.float32, initializer=g_[6])
    gb_fc3 = tf.get_variable('gb_fc3' + str(callCnt), dtype=tf.float32, initializer=g_[7])
    return [gW_conv1, gb_conv1, gW_conv2, gb_conv2, gW_fc1, gb_fc1, gW_fc3, gb_fc3]
    
def loss_svm(w_, x_shape_, x, y, callCnt=0):
    if w_ == None:
        x_shape = [ x_shape_[j] for j in range(1, len(x_shape_)) ] + [1]
        A = tf.get_variable('A' + str(callCnt), shape=x_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
        b = tf.get_variable('b' + str(callCnt), shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
    else:
        A = tf.get_variable('A' + str(callCnt), dtype=tf.float32, initializer=w_[0])
        b = tf.get_variable('b' + str(callCnt), dtype=tf.float32, initializer=w_[1])
    
    # formula (y = ax - b)
    formula = tf.matmul(x, A) - b
    
    # Loss = summation(max(0, 1 - pred*actual)) + alpha * L2_norm(A)^2
    y_reshaped = tf.reshape(y, [-1, 1]) # rank 증가시켜서 맞추기
    loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - formula * tf.cast(y_reshaped, tf.float32)))
    return loss, [A, b]

def predict_svm(w, x):
    A = w[0]
    b = w[1]
    formula = tf.matmul(x, A) - b
    y_hat = tf.squeeze(tf.cast(tf.sign(formula), tf.int32))
    return y_hat

def get_gradients_svm(loss, g_, callCnt=0):
    gA = tf.get_variable('gA' + str(callCnt), dtype=tf.float32, initializer=g_[0])
    gb = tf.get_variable('gb' + str(callCnt), dtype=tf.float32, initializer=g_[1])
    return [gA, gb]

def get_logits_2nn(x, W_0, b_0, W_1, b_1, W_2, b_2):
    hidden_output_0 = tf.nn.relu(tf.matmul(x, W_0) + b_0)
    hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0, W_1) + b_1)
    logits = tf.matmul(hidden_output_1, W_2) + b_2
    return logits

def loss_2nn(w_, x_shape_, x, y, callCnt=0):
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
    logits = get_logits_2nn(x, W_0, b_0, W_1, b_1, W_2, b_2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
    return loss, [W_0, b_0, W_1, b_1, W_2, b_2]

def predict_2nn(w, x):
    logits = get_logits_2nn(x, w[0], w[1], w[2], w[3], w[4], w[5])
    y_prob = tf.nn.softmax(logits)
    y_hat = tf.squeeze(tf.cast(tf.argmax(y_prob, 1), tf.int32))
    return y_hat

def get_gradients_2nn(loss, g_, callCnt=0):
    gW_0 = tf.get_variable('W_0' + str(callCnt), dtype=tf.float32, initializer=g_[0])
    gb_0 = tf.get_variable('b_0' + str(callCnt), dtype=tf.float32, initializer=g_[1])
    gW_1 = tf.get_variable('W_1' + str(callCnt), dtype=tf.float32, initializer=g_[2])
    gb_1 = tf.get_variable('b_1' + str(callCnt), dtype=tf.float32, initializer=g_[3])
    gW_2 = tf.get_variable('W_2' + str(callCnt), dtype=tf.float32, initializer=g_[4])
    gb_2 = tf.get_variable('b_2' + str(callCnt), dtype=tf.float32, initializer=g_[5])
    return [gW_0, gb_0, gW_1, gb_1, gW_2, gb_2]

def next_batch(batchSize, dataBatch):
    x = dataBatch['x']
    y = dataBatch['y']
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x_randBatch = [x[i] for i in idx[:batchSize]]
    y_randBatch = [y[i] for i in idx[:batchSize]]
    return x_randBatch, y_randBatch

def local_train(w_, dataBatch_i_, lr_, tau1_):
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=config, graph=graph) as sess:
            x_shape_ = dataBatch_i_['x'].shape
            x_shape = [None] + [ x_shape_[j] for j in range(1, len(x_shape_)) ]
            x = tf.placeholder(name='x', shape=x_shape, dtype=tf.float32)
            y = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
            if fl_param.MODEL_NAME == 'sr':
                (loss, w) = loss_sr(w_, x_shape_, x, y)
            elif fl_param.MODEL_NAME == 'cnn':
                if fl_param.DATA_NAME == 'mnist-o' or fl_param.DATA_NAME == 'mnist-f':
                    (loss, w) = loss_cnn(w_, x_shape_, x, y)
                elif fl_param.DATA_NAME == 'cifar10':
                    (loss, w) = loss_cnn_cifar10(w_, x_shape_, x, y)
                else:
                    raise Exception(fl_param.MODEL_NAME, fl_param.DATA_NAME)
            elif fl_param.MODEL_NAME == '2nn':
                (loss, w) = loss_2nn(w_, x_shape_, x, y)
            elif fl_param.MODEL_NAME == 'svm':
                (loss, w) = loss_svm(w_, x_shape_, x, y)
            else:
                raise Exception(fl_param.MODEL_NAME)
            # 직접 gradients 계산하는 대신 Optimizer 에서 계산된 gradients 사용
            lr = tf.placeholder(name='lr', shape=[], dtype=tf.float32)
            train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
            
            sess.run(tf.global_variables_initializer())
            w_byTime = []
            if not fl_param.SGD_ENABLED:
                for _ in range(tau1_):
                    sess.run(train_op, feed_dict={lr: lr_, x: dataBatch_i_['x'], y: dataBatch_i_['y']})
                    w_ = sess.run(w, feed_dict={lr: lr_, x: dataBatch_i_['x'], y: dataBatch_i_['y']})
                    w_byTime.append(w_)
        #             loss_ = sess.run(loss, feed_dict={lr: lr_, x: dataBatch_i_['x'], y: dataBatch_i_['y']})
        #             print(lr_, loss_)
                    lr_ *= fl_param.LR_DECAY_RATE
            else:
                numEpochSamples = dataBatch_i_['x'].shape[0]
                numItersPerEpoch = int(np.ceil(numEpochSamples / fl_param.BATCH_SIZE).tolist())
                for t1 in range(tau1_):
                    for it in range(numItersPerEpoch):
                        (sampleBatch_x, sampleBatch_y) = next_batch(fl_param.BATCH_SIZE, dataBatch_i_)
                        sess.run(train_op, feed_dict={lr: lr_, x: sampleBatch_x, y: sampleBatch_y})
                    # Epoch 마다 정보 저장
                    w_ = sess.run(w, feed_dict={lr: lr_, x: sampleBatch_x, y: sampleBatch_y})
                    w_byTime.append(w_)
        #             vs = sess.run(tf.concat([ tf.reshape(layer, [-1]) for layer in tf.get_collection(tf.GraphKeys.VARIABLES) ], axis=0))
        #             numVars = vs.shape ; print(numVars)
                    lr_ *= fl_param.LR_DECAY_RATE
    return w_byTime, w_

def federated_train_collect(w, trainData_byNid, lr, tau1):
    w_byTime_byNid = [] ; w_last_byNid = []
    for dataBatch_i in trainData_byNid:
        (w_byTime, w_last) = local_train(w, dataBatch_i, lr, tau1)
        w_byTime_byNid.append(w_byTime)
        w_last_byNid.append(w_last)
    return w_byTime_byNid, w_last_byNid

def average_w(w_byNid, weight_byNid):
    weight_byNid = np.array(weight_byNid) / sum(weight_byNid)
    w_avg = []
    for col in range(w_byNid.shape[1]):
        sum_w = np.zeros(w_byNid[0, col].shape, dtype=np.float32)
        for row in range(w_byNid.shape[0]):
            sum_w += w_byNid[row, col] * weight_byNid[row]
        w_avg.append(sum_w)
    return w_avg

def federated_aggregate(w_byTime_byNid, weight_byNid):
    w_byTime_byNid = np.array(w_byTime_byNid)
    w_byTime = [ average_w(w_byTime_byNid[:,t], weight_byNid) for t in range(len(w_byTime_byNid[0])) ]
    return w_byTime

def federated_train(w, trainData_byNid, lr, tau1, weight_byNid):
    (w_byTime_byNid, w_last_byNid) = federated_train_collect(w, trainData_byNid, lr, tau1)
    w_byTime = federated_aggregate(w_byTime_byNid, weight_byNid)
    return w_byTime, w_last_byNid

def evaluate(w_, dataBatch_):
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=config, graph=graph) as sess:
            x_shape_ = dataBatch_['x'].shape
            x_shape = [None] + [ x_shape_[j] for j in range(1, len(x_shape_)) ]
            x = tf.placeholder(name='x', shape=x_shape, dtype=tf.float32)
            y = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
            if fl_param.MODEL_NAME == 'sr':
                (loss, w) = loss_sr(w_, x_shape_, x, y)
                y_hat = predict_sr(w, x)
            elif fl_param.MODEL_NAME == 'cnn':
                if fl_param.DATA_NAME == 'mnist-o' or fl_param.DATA_NAME == 'mnist-f':
                    (loss, w) = loss_cnn(w_, x_shape_, x, y)
                    y_hat = predict_cnn(w, x)
                elif fl_param.DATA_NAME == 'cifar10':
                    (loss, w) = loss_cnn_cifar10(w_, x_shape_, x, y)
                    y_hat = predict_cnn_cifar10(w, x)
                else:
                    raise Exception(fl_param.MODEL_NAME, fl_param.DATA_NAME)
            elif fl_param.MODEL_NAME == '2nn':
                (loss, w) = loss_2nn(w_, x_shape_, x, y)
                y_hat = predict_2nn(w, x)
            elif fl_param.MODEL_NAME == 'svm':
                (loss, w) = loss_svm(w_, x_shape_, x, y)
                y_hat = predict_svm(w, x)
            else:
                raise Exception(fl_param.MODEL_NAME)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(y_hat, y), tf.float32))
            
            sess.run(tf.global_variables_initializer())

            # Batch Size 만큼 나눠서 평가
            numTestIters = min(fl_param.NUM_TEST_ITERS, int(len(dataBatch_['x'])/fl_param.BATCH_SIZE))
            losses_ = [] ; accs_ = [] ; idxBegin = 0 ; idxEnd = 0
            for i in range(numTestIters):
                idxEnd += fl_param.BATCH_SIZE
                sampleBatch_x = dataBatch_['x'][idxBegin:idxEnd]
                sampleBatch_y = dataBatch_['y'][idxBegin:idxEnd]
                (loss_, acc_) = sess.run((loss, accuracy), feed_dict={x: sampleBatch_x, y: sampleBatch_y})
                losses_.append(loss_)
                accs_.append(acc_)
                idxBegin = idxEnd
    return np.mean(losses_), np.mean(accs_)

def local_gradients(w_, dataBatch_):
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=config, graph=graph) as sess:
            x_shape_ = dataBatch_['x'].shape
            x_shape = [None] + [ x_shape_[j] for j in range(1, len(x_shape_)) ]
            x = tf.placeholder(name='x', shape=x_shape, dtype=tf.float32)
            y = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
            if fl_param.MODEL_NAME == 'sr':
                (loss, w) = loss_sr(w_, x_shape_, x, y)
            elif fl_param.MODEL_NAME == 'cnn':
                if fl_param.DATA_NAME == 'mnist-o' or fl_param.DATA_NAME == 'mnist-f':
                    (loss, w) = loss_cnn(w_, x_shape_, x, y)
                elif fl_param.DATA_NAME == 'cifar10':
                    (loss, w) = loss_cnn_cifar10(w_, x_shape_, x, y)
                else:
                    raise Exception(fl_param.MODEL_NAME, fl_param.DATA_NAME)
            elif fl_param.MODEL_NAME == '2nn':
                (loss, w) = loss_2nn(w_, x_shape_, x, y)
            elif fl_param.MODEL_NAME == 'svm':
                (loss, w) = loss_svm(w_, x_shape_, x, y)
            else:
                raise Exception(fl_param.MODEL_NAME)
            g = tf.gradients(loss, w)
            
            sess.run(tf.global_variables_initializer())
            g_ = sess.run(g, feed_dict={x: dataBatch_['x'], y: dataBatch_['y']})
    return g_

def federated_collect_gradients(w, nid2_Data_i):
    gs = [] ; nid2_gs = {}
    for nid in nid2_Data_i.keys():
        g = local_gradients(w, nid2_Data_i[nid])
        g = np_flatten(g)
        gs.append(g)
        nid2_gs[nid] = g
    return gs, nid2_gs

def np_flatten(model):
    model_ = []
    for i in range(len(model)):
        model_ += model[i].flatten().tolist()
    return np.array(model_)