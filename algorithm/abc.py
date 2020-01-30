import importlib
from wurlitzer import pipes

import numpy as np
import os
import csv
from time import gmtime, strftime #strftime("%m%d_%H%M%S", gmtime()) + ' ' + 
import collections

from abc import ABC, abstractmethod

from cloud.cloud import Cloud
from visualization.util import saveGraphOfLog
import fl_const
import fl_data
import fl_util

PRINT_INTERVAL = 0.5

class AbstractAlgorithm(ABC):
    
    @abstractmethod
    def getFileName(self):
        pass
    
    def __init__(self, args):
        self.args = args
        
        self.trainData_by1Nid = fl_util.deserialize(os.path.join(fl_const.DATA_DIR_PATH, args.dataName, 'train'))
        if args.isValidation == True:
            self.testData_by1Nid = fl_util.deserialize(os.path.join(fl_const.DATA_DIR_PATH, args.dataName, 'val'))
        else:
            self.testData_by1Nid = fl_util.deserialize(os.path.join(fl_const.DATA_DIR_PATH, args.dataName, 'test'))
            
        c_log_file = open(os.path.join(fl_const.LOG_DIR_PATH, fl_const.C_LOG_FILE_NAME), 'a')
        with pipes(stdout=c_log_file, stderr=c_log_file):
            modelPackagePath = 'model.' + args.modelName
            modelModule = importlib.import_module(modelPackagePath)
            Model = getattr(modelModule, 'Model')
            self.model = Model(args, self.trainData_by1Nid, self.testData_by1Nid)
        c_log_file.close()
        
        fileNameBase = self.getFileName()
        print(fileNameBase)
        self.fileEpoch = open(os.path.join(fl_const.LOG_DIR_PATH, fileNameBase + '_' + fl_const.EPOCH_CSV_POSTFIX), 'w', newline='', buffering=1)
        self.fwEpoch = csv.writer(self.fileEpoch, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        (trainData_byNid, z_edge) = fl_data.groupByEdge(self.trainData_by1Nid, args.nodeType, args.edgeType, args.numNodes, args.numEdges)
#         print('Num Data per Node:', [ len(D_i['x']) for D_i in trainData_byNid ])
#         numClassPerNodeList = [ len(np.unique(D_i['y'])) for D_i in trainData_byNid ]
#         print('Num Class per Node:', numClassPerNodeList)
#         counter = collections.Counter(numClassPerNodeList)
#         print('Num Class per Node Counter:', counter)
        print('Shape of trainData on 1st node:', trainData_byNid[0]['x'].shape, trainData_byNid[0]['y'].shape)
        
        topologyPackagePath = 'cloud.' + args.topologyName
        topologyModule = importlib.import_module(topologyPackagePath)
        Topology = getattr(topologyModule, 'Topology')
        topology = Topology(args.numNodes, args.numEdges)
        self.c = Cloud(topology, trainData_byNid, args.numEdges)
        nids_byGid = fl_data.to_nids_byGid(z_edge)
        self.c.digest(nids_byGid)
        # Note that the cloud c is initialized with numEdges instead of numGroups
        # because the concept of groups is not considered by most of algorithms
        # But, for those algorithms that care about the concept of groups like ch-fedavg,
        # the cloud c needs to be re-initialized with different numGroups
        
        print('Profiling Delays...')
        self.speed2Delay = {}
        for linkSpeed in args.linkSpeeds:
            d_group = self.getCommTimeGroup(self.c, self.model.size, linkSpeed)
            d_global = self.getCommTimeGlobal(self.c, self.model.size, linkSpeed)
            print('linkSpeed:', linkSpeed, ', d_group:', d_group, ', d_global:', d_global)
            self.speed2Delay[linkSpeed] = (d_group, d_global)
            
        print('Default Delay...')
        d_local, d_group, d_global = self.getDefaultDelay()
        print('d_local:', d_local, ', d_group:', d_group, ', d_global:', d_global)
        print()
    
    @abstractmethod
    def run(self):
        pass
    
    def finalize(self):
        self.fileEpoch.close()
        print()
        
        print('Dump JSON...')
        totalComps = self.epoch * self.getFlopOpsPerEpoch()
        totalComms = self.getTotalComms()
        d_local, d_group, d_global = self.getDefaultDelay()
        metadata = { 'args': vars(self.args),
                    'modelFlopOps': self.model.flopOps,
                    'modelSize': self.model.size,
                    'numTrainSamples': len(self.trainData_by1Nid[0]['x']),
                    'numTestSamples': len(self.testData_by1Nid[0]['x']),
                    'totalComps': totalComps,
                    'totalComms': totalComms,
                    'd_local': d_local,
                    'd_group': d_group,
                    'd_global': d_global
                   }
        fileNameBase = self.getFileName()
        fl_util.dumpJson(os.path.join(fl_const.LOG_DIR_PATH, fileNameBase + '.json'), metadata)
        
        print('Dump Graph...')
        filePathEpoch = os.path.join(fl_const.LOG_DIR_PATH, fileNameBase + '_' + fl_const.EPOCH_CSV_POSTFIX)
        saveGraphOfLog(filePathEpoch, 0, 2)
        
        # remove all the previous time logs manually
        for fileName in os.listdir(fl_const.LOG_DIR_PATH):
            if fileName.startswith(fileNameBase) and fileName.endswith(fl_const.TIME_CSV_POSTFIX):
                os.remove(os.path.join(fl_const.LOG_DIR_PATH, fileName))
                
        print('Dump Time Logs...')
        for procSpeed in self.args.procSpeeds:
            for linkSpeed in self.args.linkSpeeds:
                self.dumpTimeLogs(procSpeed, linkSpeed)
        print()
                
    def getDefaultDelay(self):
        default_procSpeed = self.args.procSpeeds[0]
        default_linkSpeed = self.args.linkSpeeds[0]
        d_local = self.getFlopOpsPerEpoch() / (default_procSpeed*1e9) # GHz
        d_group, d_global = self.speed2Delay[default_linkSpeed]
        return d_local, d_group, d_global
    
    def setDefaultCommDelay(self, d_group_new, d_global_new):
        _, d_group_prev, d_global_prev = self.getDefaultDelay()
        d_group_ratio = d_group_new / d_group_prev
        d_global_ratio = d_global_new / d_global_prev
        
        # Scale all the previous delays
        for linkSpeed in self.speed2Delay.keys():
            d_group, d_global = self.speed2Delay[linkSpeed]
            self.speed2Delay[linkSpeed] = (d_group*d_group_ratio, d_global*d_global_ratio)
            
    def getFlopOpsPerEpoch(self):
        # Largest Flop Ops Idea: https://github.com/TalwalkarLab/leaf/blob/master/models/metrics/visualization_utils.py#L263
        lenLargestData = max([ len(self.c.get_nid2_D_i()[nid]['x']) for nid in range(len(self.c.get_N())) ])
        
        # Equation for Calculating Flop Ops: https://github.com/TalwalkarLab/leaf/blob/master/models/model.py#L91
        return lenLargestData * self.model.flopOps
    
    def getApprCommCostGroup(self, c):
        return 0
        
    def getApprCommCostGlobal(self, c):
        return 0
        
    def getCommTimeGroup(self, c, dataSize, linkSpeed):
        return 0
        
    def getCommTimeGlobal(self, c, dataSize, linkSpeed):
        return 0
        
    def getTotalComms(self):
        if self.args.algName == 'cgd': return 0
        
        fileNameBase = self.getFileName()
        fileEpoch = open(os.path.join(fl_const.LOG_DIR_PATH, fileNameBase + '_' + fl_const.EPOCH_CSV_POSTFIX), 'r')
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
        return cntGroupComms * self.getApprCommCostGroup(self.c) + cntGlobalComms * self.getApprCommCostGlobal(self.c)
    
    def dumpTimeLogs(self, procSpeed, linkSpeed):
        d_local = self.getFlopOpsPerEpoch() / (procSpeed*1e9) # GHz
        d_group, d_global = self.speed2Delay[linkSpeed]
        print('d_local:', d_local, ', d_group:', d_group, ', d_global:', d_global)
        
        # get accuracy in advance
        fileNameBase = self.getFileName()
        fileEpoch = open(os.path.join(fl_const.LOG_DIR_PATH, fileNameBase + '_' + fl_const.EPOCH_CSV_POSTFIX), 'r')
        lines = fileEpoch.readlines()
        tokens = lines[-1].rstrip('\n').split(',')
        accuracy = tokens[2]
        
        # create time log file
        fileEpoch = open(os.path.join(fl_const.LOG_DIR_PATH, fileNameBase + '_' + fl_const.EPOCH_CSV_POSTFIX), 'r')
        fileEpoch.readline() # ignore the first title line
        
        fileNameTime = '%s_%d_%d_%.3f_%s' % (fileNameBase, procSpeed, linkSpeed, float(accuracy), fl_const.TIME_CSV_POSTFIX)
        print(fileNameTime)
        fileTime = open(os.path.join(fl_const.LOG_DIR_PATH, fileNameTime), 'w', newline='', buffering=1)
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
            # 기존에 값이 있고, 기존 aggrType 이 'Group' 또는 'Global' 일 경우 시간 하나 미루기
            if nextTime in refDict and (refDict[nextTime][3] == 'Group' or refDict[nextTime][3] == 'Global'):
                nextTime += PRINT_INTERVAL
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