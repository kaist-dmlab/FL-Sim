import tensorflow as tf
import importlib
import numpy as np
import csv
from time import gmtime, strftime #strftime("%m%d_%H%M%S", gmtime()) + ' ' + 

from abc import ABC, abstractmethod

import fl_data
import fl_struct

PRINT_INTERVAL = 0.5

def printTimedLogs(fileName):
    fileEpoch = open('logs/' + fileName + '_epoch.csv', 'r')
    fileEpoch.readline() # 세 줄 제외
    fileEpoch.readline()
    fileEpoch.readline()
    fileTime = open('logs/' + fileName + '_time.csv', 'w', newline='', buffering=1)
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
    
    @abstractmethod
    def getFileName(self):
        pass
    
    def __init__(self, args):
        self.args = args
        if self.args.dataName == 'mnist-o':
            trainData, testData = tf.keras.datasets.mnist.load_data()
        elif self.args.dataName == 'mnist-f':
            trainData, testData = tf.keras.datasets.fashion_mnist.load_data()
        elif self.args.dataName == 'cifar10':
            trainData, testData = tf.keras.datasets.cifar10.load_data()
        else:
            raise Exception(self.args.dataName)
            
        self.trainData_by1Nid = fl_data.preprocess(self.args.modelName, self.args.dataName, trainData, self.args.flatten)
        self.testData_by1Nid = fl_data.preprocess(self.args.modelName, self.args.dataName, testData, self.args.flatten)
        (trainData_byNid, train_z) = fl_data.groupByEdge(self.args.modelName, self.args.dataName, trainData,
                                                          self.args.nodeType, self.args.edgeType, self.args.numNodes, self.args.numEdges, self.args.flatten)
        print('Shape of trainData on 1st node:', trainData_byNid[0]['x'].shape, trainData_byNid[0]['y'].shape)
        
        modelPackagePath = 'model.' + self.args.modelName
        modelModule = importlib.import_module(modelPackagePath)
        Model = getattr(modelModule, 'Model')
        self.model = Model(self.args, trainData_byNid[0]['x'].shape)
        
        fileName = self.getFileName()
        print(fileName)
        self.fileEpoch = open('logs/' + fileName + '_epoch.csv', 'w', newline='', buffering=1)
        self.fwEpoch = csv.writer(self.fileEpoch, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.fwEpoch.writerow([ 'NUM_NODES', 'NUM_EDGES', 'MAX_TIME', 'SGD_ENABLED', 'MODEL_SIZE', \
                               'LR_INITIAL', 'LR_DECAY_RATE', 'NUM_TEST_SAMPLES', 'BATCH_SIZE' ])
        self.fwEpoch.writerow([ args.numNodes, args.numEdges, args.maxTime, args.sgdEnabled, self.model.size, \
                     args.lrInitial, args.lrDecayRate, args.numTestSamples, args.batchSize ])
        
        ft = fl_struct.FatTree(self.args.numNodes, self.args.numEdges)
        self.c = fl_struct.Cloud(ft, trainData_byNid, self.args.numEdges, self.model.size)
        self.c.digest(train_z)
        
    def __del__(self):
        print()
        self.fileEpoch.close()
        fileName = self.getFileName()
        printTimedLogs(fileName)
        
    def getInitVars(self):
        return self.trainData_by1Nid, self.testData_by1Nid, self.c
    
    @abstractmethod
    def run(self):
        pass