import tensorflow as tf
import numpy as np
import csv
from time import gmtime, strftime #strftime("%m%d_%H%M%S", gmtime()) + ' ' + 

from abc import ABC, abstractmethod

import fl_param
import fl_data
import fl_struct

PRINT_INTERVAL = 0.5

def printTimedLogs(commonFileName):
    fileEpoch = open('logs/' + commonFileName + '_epoch.csv', 'r')
    fileEpoch.readline() # 세 줄 제외
    fileEpoch.readline()
    fileEpoch.readline()
    fileTime = open('logs/' + commonFileName + '_time.csv', 'w', newline='', buffering=1)
    fwTime = csv.writer(fileTime, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    fwTime.writerow(['time', 'loss', 'accuracy', 'epoch', 'aggrType'])
    refDict = {}
    
    # 시간 0일 때는 그래프 출력용 초기값 설정을 입력하기 위해, Epoch 1 의 값을 가져옴
    line = fileEpoch.readline()
    if not line: return
    tokens = line.split(',')
    epoch = tokens[0] ; loss = tokens[1] ; accuracy = tokens[2]
    refDict[0] = [loss, accuracy, epoch, '']
    while True:
        line = fileEpoch.readline()
        if not line: break
        line = line.rstrip('\n')
        tokens = line.split(',')
        epoch = tokens[0] ; loss = tokens[1] ; accuracy = tokens[2] ; time = tokens[3] ; aggrType = tokens[4]
        nextTime = np.ceil(float(time)/PRINT_INTERVAL)*PRINT_INTERVAL
        if not(nextTime in refDict):
            refDict[nextTime] = [loss, accuracy, epoch, aggrType]
    maxRefDict = max(refDict)
    for i in range(100000): # 최대 100000 Time Interval 측정
        curTime = i * PRINT_INTERVAL
        if curTime > maxRefDict: break
        for j in reversed(range(i+1)):
            prevTime = j * PRINT_INTERVAL
            if prevTime in refDict: break
        [loss, accuracy, epoch, aggrType] = refDict[prevTime]
        fwTime.writerow([curTime, loss, accuracy, epoch, aggrType])
    fileTime.close()
    fileEpoch.close()
    
class AbstractAlgorithm(ABC):
    
    TITLE_PARAMS = str(fl_param.NUM_NODES) + '_' + str(fl_param.NUM_EDGES) + '_' + fl_param.MODEL_NAME + '_' \
            + fl_param.DATA_NAME
    
    def __init__(self, args):
        self.args = args
        
        commonFileName = self.TITLE_PARAMS + '_' + self.getName()
        print(commonFileName)
        self.fileEpoch = open('logs/' + commonFileName + '_epoch.csv', 'w', newline='', buffering=1)
        self.fwEpoch = csv.writer(self.fileEpoch, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.fwEpoch.writerow([ 'MAX_TIME', 'SGD_ENABLED', 'MODEL_SIZE', 'LR_INITIAL', 'LR_DECAY_RATE', 'BATCH_SIZE', 'NUM_TEST_ITERS' ])
        self.fwEpoch.writerow([ fl_param.MAX_TIME, fl_param.SGD_ENABLED, fl_param.MODEL_SIZE, \
                     fl_param.LR_INITIAL, fl_param.LR_DECAY_RATE, fl_param.BATCH_SIZE, fl_param.NUM_TEST_ITERS ])
        
    def __del__(self):
        print()
        self.fileEpoch.close()
        commonFileName = self.TITLE_PARAMS + '_' + self.getName()
        printTimedLogs(commonFileName)
        
    @abstractmethod
    def getName(self):
        pass
        
    def initialize(self, nodeType, edgeType):
        if fl_param.DATA_NAME == 'mnist-o':
            trainData, testData = tf.keras.datasets.mnist.load_data()
        elif fl_param.DATA_NAME == 'mnist-f':
            trainData, testData = tf.keras.datasets.fashion_mnist.load_data()
        elif fl_param.DATA_NAME == 'cifar10':
            trainData, testData = tf.keras.datasets.cifar10.load_data()
        else:
            raise Exception(fl_param.DATA_NAME)

        flatten = False if fl_param.MODEL_NAME == 'cnn' else True
        trainData_by1Nid = fl_data.preprocess(fl_param.MODEL_NAME, fl_param.DATA_NAME, trainData, flatten)
        testData_by1Nid = fl_data.preprocess(fl_param.MODEL_NAME, fl_param.DATA_NAME, testData, flatten)
        (trainData_byNid, train_z) = fl_data.groupByEdge(fl_param.MODEL_NAME, fl_param.DATA_NAME, trainData,
                                                          nodeType, edgeType, fl_param.NUM_NODES, fl_param.NUM_EDGES, flatten)
        (trainData_byNid_iid, train_z_iid) = fl_data.groupByEdge(fl_param.MODEL_NAME, fl_param.DATA_NAME, trainData,
                                                          nodeType, 'all', fl_param.NUM_NODES, fl_param.NUM_EDGES, flatten)
        print('Shape of trainData on 1st node:', trainData_byNid[0]['x'].shape, trainData_byNid[0]['y'].shape)

        ft = fl_struct.FatTree(fl_param.NUM_NODES, fl_param.NUM_EDGES)
        c = fl_struct.Cloud(ft, trainData_byNid, fl_param.NUM_EDGES)
        c.digest(train_z)
        c2 = fl_struct.Cloud(ft, trainData_byNid_iid, fl_param.NUM_EDGES)
        c2.digest(train_z_iid)
        return trainData_by1Nid, testData_by1Nid, c, c2
    
    @abstractmethod
    def run(self):
        pass