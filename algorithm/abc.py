import tensorflow as tf
import importlib
import numpy as np
import os
import collections
import csv
from time import gmtime, strftime #strftime("%m%d_%H%M%S", gmtime()) + ' ' + 

from abc import ABC, abstractmethod

import fl_data
import fl_struct
import fl_util

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
    
def groupRandomly(numNodes, numGroups):
    numNodesPerGroup = int(numNodes / numGroups)
    z_rand = [ k for k in range(numGroups) for _ in range(numNodesPerGroup) ]
    np.random.shuffle(z_rand)
    return z_rand

class AbstractAlgorithm(ABC):
    
    @abstractmethod
    def getFileName(self):
        pass
    
    def __init__(self, args, randomEnabled=False):
        self.args = args
        
        self.trainData_by1Nid = fl_util.deserialize(os.path.join('data', self.args.dataName, 'train'))
        if self.args.isValidation == True:
            self.testData_by1Nid = fl_util.deserialize(os.path.join('data', self.args.dataName, 'val'))
        else:
            self.testData_by1Nid = fl_util.deserialize(os.path.join('data', self.args.dataName, 'test'))
        
        modelPackagePath = 'model.' + self.args.modelName
        modelModule = importlib.import_module(modelPackagePath)
        Model = getattr(modelModule, 'Model')
        self.model = Model(self.args, self.trainData_by1Nid, self.testData_by1Nid)
        
        (trainData_byNid, z_edge) = fl_data.groupByEdge(self.trainData_by1Nid, self.args.numNodeClasses, self.args.numEdgeClasses, self.args.numNodes, self.args.numEdges)
#         print('Num Data per Node:', [ len(D_i['x']) for D_i in trainData_byNid ])
#         numClassPerNodeList = [ len(np.unique(D_i['y'])) for D_i in trainData_byNid ]
#         print('Num Class per Node:', numClassPerNodeList)
#         counter = collections.Counter(numClassPerNodeList)
#         print('Num Class per Node Counter:', counter)
        print('Shape of trainData on 1st node:', trainData_byNid[0]['x'].shape, trainData_byNid[0]['y'].shape)
        
        fileName = self.getFileName()
        print(fileName)
        self.fileEpoch = open('logs/' + fileName + '_epoch.csv', 'w', newline='', buffering=1)
        self.fwEpoch = csv.writer(self.fileEpoch, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.fwEpoch.writerow([ 'NUM_NODES', 'NUM_EDGES', 'MAX_TIME', 'SGD_ENABLED', 'MODEL_SIZE', \
                               'LR_INITIAL', 'LR_DECAY_RATE', 'NUM_TEST_SAMPLES', 'BATCH_SIZE' ])
        self.fwEpoch.writerow([ args.numNodes, args.numEdges, args.maxTime, args.sgdEnabled, self.model.size, \
                     args.lrInitial, args.lrDecayRate, args.numTestSamples, args.batchSize ])
        
        ft = fl_struct.FatTree(self.args.numNodes, self.args.numEdges)
        self.c = fl_struct.Cloud(ft, trainData_byNid, self.args.numGroups, self.model.size)
        if randomEnabled == False:
            self.c.digest(z_edge)
        else:
            z_rand = groupRandomly(self.args.numNodes, self.args.numGroups)    
            self.c.digest(z_rand)
        
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