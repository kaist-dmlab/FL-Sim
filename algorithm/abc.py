import importlib
import numpy as np
import os
import collections
import csv
from time import gmtime, strftime #strftime("%m%d_%H%M%S", gmtime()) + ' ' + 

from abc import ABC, abstractmethod

from cloud.cloud import Cloud
import fl_data
import fl_util

LOG_DIR_NAME = 'log'
EPOCH_CSV_POSTFIX = 'epoch.csv'
TIME_CSV_POSTFIX = 'time.csv'

PRINT_INTERVAL = 0.5
    
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
        
        (trainData_byNid, z_edge) = fl_data.groupByEdge(self.trainData_by1Nid, self.args.nodeType, self.args.edgeType, self.args.numNodes, self.args.numEdges)
#         print('Num Data per Node:', [ len(D_i['x']) for D_i in trainData_byNid ])
#         numClassPerNodeList = [ len(np.unique(D_i['y'])) for D_i in trainData_byNid ]
#         print('Num Class per Node:', numClassPerNodeList)
#         counter = collections.Counter(numClassPerNodeList)
#         print('Num Class per Node Counter:', counter)
        print('Shape of trainData on 1st node:', trainData_byNid[0]['x'].shape, trainData_byNid[0]['y'].shape)
        
        fileName = self.getFileName()
        print(fileName)
        self.fileEpoch = open(os.path.join(LOG_DIR_NAME, fileName + '_' + EPOCH_CSV_POSTFIX), 'w', newline='', buffering=1)
        self.fwEpoch = csv.writer(self.fileEpoch, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        topologyPackagePath = 'cloud.' + self.args.topologyName
        topologyModule = importlib.import_module(topologyPackagePath)
        Topology = getattr(topologyModule, 'Topology')
        topology = Topology(self.model.size, self.args.numNodes, self.args.numEdges)
        self.c = Cloud(topology, trainData_byNid, self.args.numGroups)
        if randomEnabled == False:
            self.c.digest(z_edge)
        else:
            z_rand = groupRandomly(self.args.numNodes, self.args.numGroups)
            self.c.digest(z_rand)
        
        self.groupTimeMap = {}
        self.globalTimeMap = {}
        for b_group in [ 10 ]:
            d_group = self.getCommTimeGroupExt(b_group)
            self.groupTimeMap[b_group] = d_group
        for b_global in [ 10 ]:
            d_global = self.getCommTimeGlobalExt(b_global)
            self.globalTimeMap[b_global] = d_global
    
    @abstractmethod
    def run(self):
        pass
    
    def __del__(self):
        self.fileEpoch.close()
        print()
        
        # Largest Flop Ops Idea: https://github.com/TalwalkarLab/leaf/blob/master/models/metrics/visualization_utils.py#L263
        if self.args.algName == 'cgd':
            lenLargestData = len(self.trainData_by1Nid[0]['x'])
        else:
            lenLargestData = max([ len(self.c.get_nid2_D_i()[nid]['x']) for nid in range(len(self.c.get_N())) ])
        
        # Equation for Calculating Flop Ops: https://github.com/TalwalkarLab/leaf/blob/master/models/model.py#L91
        epochFlopOps = lenLargestData * self.model.flopOps
        totalComps = self.t * epochFlopOps
        totalComms = self.getTotalComms()
        metadata = { 'args': vars(self.args),
                    'modelFlopOps': self.model.flopOps,
                    'modelSize': self.model.size,
                    'numTrainSamples': len(self.trainData_by1Nid[0]['x']),
                    'numTestSamples': len(self.testData_by1Nid[0]['x']),
                    'totalComps': totalComps,
                    'totalComms': totalComms
                   }
        fileName = self.getFileName()
        fl_util.dumpJson(os.path.join(LOG_DIR_NAME, fileName + '.json'), metadata)
        
        if self.args.algName != 'cgd':
            b_local = 1
            b_group = 10
            b_global = 10
            self.dumpTimeLogs(epochFlopOps, b_local, b_group, b_global)
        
    def getApprCommCostGroup(self):
        return 0
        
    def getApprCommCostGlobal(self):
        return 0
        
    def getCommTimeGroupExt(self, b_group):
        b_group = str(int(b_group/10)) + 'MBps' # 10: cloud.fattree 참조
        return self.getCommTimeGroup(b_group)
        
    def getCommTimeGroup(self, b_group):
        return 0
        
    def getCommTimeGlobalExt(self, b_global):
        b_global = str(int(b_global/10)) + 'MBps'
        return self.getCommTimeGlobal(b_global)
        
    def getCommTimeGlobal(self, b_global):
        return 0
        
    def getTotalComms(self):
        if self.args.algName == 'cgd': return 0
        
        fileName = self.getFileName()
        fileEpoch = open(os.path.join(LOG_DIR_NAME, fileName + '_' + EPOCH_CSV_POSTFIX), 'r')
        line = fileEpoch.readline() # 제목 줄 제외
        tokens = line.rstrip('\n').split(',')
        if tokens[3] != 'aggrType': raise Exception(line)
            
        cntGroupComms = 0 ; cntGlobalComms = 0
        while True:
            line = fileEpoch.readline()
            if not line: break
            tokens = line.rstrip('\n').split(',')
            epoch = tokens[0] ; loss = tokens[1] ; accuracy = tokens[2] ; aggrType = tokens[3]
            if aggrType == 'Group':
                cntGroupComms += 1
            if aggrType == 'Global':
                cntGlobalComms += 1
        return cntGroupComms * self.getApprCommCostGroup() + cntGlobalComms * self.getApprCommCostGlobal()
    
    def dumpTimeLogs(self, epochFlopOps, b_local, b_group, b_global):
        d_local = epochFlopOps / (b_local*1000000000) # GHz
        
        d_group = self.groupTimeMap[b_group]
        d_global = self.globalTimeMap[b_global]
        
        fileName = self.getFileName()
        fileEpoch = open(os.path.join(LOG_DIR_NAME, fileName + '_' + EPOCH_CSV_POSTFIX), 'r')
        fileEpoch.readline() # 제목 줄 제외
        fileTime = open(os.path.join(LOG_DIR_NAME, fileName + '_' + str(b_local) + '_' + str(b_group) + '_' + str(b_global) + '_' + TIME_CSV_POSTFIX), 'w', newline='', buffering=1)
        fwTime = csv.writer(fileTime, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fwTime.writerow(['time', 'loss', 'accuracy', 'epoch', 'aggrType'])
        refDict = {}
        
        # 시간 0일 때는 그래프 출력용 초기값 설정을 입력하기 위해, 첫번째 Epoch의 값을 가져옴
        line = fileEpoch.readline()
        if not line: return
        tokens = line.rstrip('\n').split(',')
        epoch = tokens[0] ; loss = tokens[1] ; accuracy = tokens[2] ; aggrType = tokens[3]
        refDict[0] = [loss, accuracy, epoch, aggrType]
        
        isFirstLine = True
        time = 0
        while True:
            if isFirstLine == True:
                isFirstLine = False
            else:
                line = fileEpoch.readline()
            if not line: break
            tokens = line.rstrip('\n').split(',')
            epoch = tokens[0] ; loss = tokens[1] ; accuracy = tokens[2] ; aggrType = tokens[3]
            if aggrType == '':
                time += d_local
            elif aggrType == 'Group':
                time += d_local + d_group
            elif aggrType == 'Global':
                time += d_local + d_global
            else:
                raise Exception(aggrType)
            nextTime = np.ceil(float(time)/PRINT_INTERVAL)*PRINT_INTERVAL
#             if not(nextTime in refDict):
            refDict[nextTime] = [loss, accuracy, epoch, aggrType]
    
        maxTimeInRefDict = max(refDict)
        for i in range(100000): # 최대 100000 Time Interval 측정
            curTime = i * PRINT_INTERVAL
            if curTime > maxTimeInRefDict: break
            for j in reversed(range(i+1)):
                prevTime = j * PRINT_INTERVAL
                if prevTime in refDict: break
            [loss, accuracy, epoch, aggrType] = refDict[prevTime]
            fwTime.writerow([curTime, loss, accuracy, epoch, aggrType])
        fileTime.close()
        fileEpoch.close()