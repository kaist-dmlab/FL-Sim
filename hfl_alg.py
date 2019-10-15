# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import csv
from time import gmtime, strftime #strftime("%m%d_%H%M%S", gmtime()) + ' ' + 

import hfl_param
import hfl_core
import hfl_util
import hfl_struct

TITLE_PARAMS = str(hfl_param.NUM_NODES) + '_' + str(hfl_param.NUM_EDGES) + '_' + hfl_param.DATA_NAME + '_' \
        + hfl_param.MODEL_NAME
def printAllParams(fw):
    fw.writerow([ 'MAX_TIME', 'SGD_ENABLED', 'MODEL_SIZE', 'LR_INITIAL', 'LR_DECAY_RATE', 'BATCH_SIZE', 'NUM_TEST_ITERS' ])
    fw.writerow([ hfl_param.MAX_TIME, hfl_param.SGD_ENABLED, hfl_param.MODEL_SIZE, \
                 hfl_param.LR_INITIAL, hfl_param.LR_DECAY_RATE, hfl_param.BATCH_SIZE, hfl_param.NUM_TEST_ITERS ])
    
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
    
def initialize(nodeType, edgeType):
    if hfl_param.DATA_NAME == 'mnist-o':
        trainData, testData = tf.keras.datasets.mnist.load_data()
    elif hfl_param.DATA_NAME == 'mnist-f':
        trainData, testData = tf.keras.datasets.fashion_mnist.load_data()
    elif hfl_param.DATA_NAME == 'cifar10':
        trainData, testData = tf.keras.datasets.cifar10.load_data()
    else:
        raise Exception(hfl_param.DATA_NAME)

    flatten = False if hfl_param.MODEL_NAME == 'cnn' else True
    trainData_by1Nid = hfl_util.preprocess(hfl_param.MODEL_NAME, hfl_param.DATA_NAME, trainData, flatten)
    testData_by1Nid = hfl_util.preprocess(hfl_param.MODEL_NAME, hfl_param.DATA_NAME, testData, flatten)
    (trainData_byNid, train_z) = hfl_util.groupByEdge(hfl_param.MODEL_NAME, hfl_param.DATA_NAME, trainData,
                                                      nodeType, edgeType, hfl_param.NUM_NODES, hfl_param.NUM_EDGES, flatten)
    (trainData_byNid_iid, train_z_iid) = hfl_util.groupByEdge(hfl_param.MODEL_NAME, hfl_param.DATA_NAME, trainData,
                                                      nodeType, 'all', hfl_param.NUM_NODES, hfl_param.NUM_EDGES, flatten)
    print('Shape of trainData on 1st node:', trainData_byNid[0]['x'].shape, trainData_byNid[0]['y'].shape)

    ft = hfl_struct.FatTree(hfl_param.NUM_NODES, hfl_param.NUM_EDGES)
    c = hfl_struct.Cloud(ft, trainData_byNid, hfl_param.NUM_EDGES)
    c.digest(train_z)
    c2 = hfl_struct.Cloud(ft, trainData_byNid_iid, hfl_param.NUM_EDGES)
    c2.digest(train_z_iid)
    return trainData_by1Nid, testData_by1Nid, c, c2
    
def CGD(nodeType, edgeType):
    commonFileName = TITLE_PARAMS + '_' + nodeType + '_' + edgeType + '_CGD'
    fileEpoch = open('logs/' + commonFileName + '_epoch.csv', 'w', newline='', buffering=1)
    fwEpoch = csv.writer(fileEpoch, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    (trainData_by1Nid, testData_by1Nid, _, _) = initialize(nodeType, edgeType)
    
    print(commonFileName)
    printAllParams(fwEpoch)
    fwEpoch.writerow(['epoch', 'loss', 'accuracy'])
    w = None ; lr = hfl_param.LR_INITIAL
    for t in range(1, hfl_param.MAX_EPOCH):
        (w_byTime, _) = hfl_core.federated_train(w, trainData_by1Nid, lr, 1, [1])
        w = w_byTime[0]
        (loss, accuracy) = hfl_core.evaluate(w, testData_by1Nid[0])
        print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f' % (t, loss, accuracy))
        fwEpoch.writerow([t, loss, accuracy])
        lr *= hfl_param.LR_DECAY_RATE
    print()
    fileEpoch.close()
    
def FedAvg(nodeType, edgeType, tau1):
    commonFileName = TITLE_PARAMS + '_' + nodeType + '_' + edgeType + '_FedAvg_' + str(tau1)
    fileEpoch = open('logs/' + commonFileName + '_epoch.csv', 'w', newline='', buffering=1)
    fwEpoch = csv.writer(fileEpoch, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    (_, testData_by1Nid, c, _) = initialize(nodeType, edgeType)
    
    print(commonFileName)
    printAllParams(fwEpoch)
    fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'time', 'aggrType'])
    d_global = c.get_d_global(False) ; d_sum = 0
    w = None ; lr = hfl_param.LR_INITIAL
#     for t2 in range(int(hfl_param.MAX_EPOCH/tau1)):
    t2 = 0
    while True:
        (w_byTime, _) = hfl_core.federated_train(w, c.get_Data_is(), lr, tau1, c.get_D_is())
        w = w_byTime[-1]
        for t1 in range(tau1):
            t = t2*tau1 + t1 + 1
            if t % tau1 == 0:
                d_sum += d_global
                aggrType = 'Global'
            else:
                aggrType = ''
            (loss, accuracy) = hfl_core.evaluate(w_byTime[t1], testData_by1Nid[0])
            print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f\ttime=%.4f\taggrType=%s' % (t, loss, accuracy, d_sum, aggrType))
            fwEpoch.writerow([t, loss, accuracy, d_sum, aggrType])
            
            lr *= hfl_param.LR_DECAY_RATE
            if d_sum >= hfl_param.MAX_TIME: break;
        if d_sum >= hfl_param.MAX_TIME: break;
        t2 += 1
    print()
    fileEpoch.close()
    printTimedLogs(commonFileName)
    
def HierFAVG(nodeType, edgeType, tau1, tau2):
    commonFileName = TITLE_PARAMS + '_' + nodeType + '_' + edgeType + '_HierFAVG_' + str(tau1) + '_' + str(tau2)
    fileEpoch = open('logs/' + commonFileName + '_epoch.csv', 'w', newline='', buffering=1)
    fwEpoch = csv.writer(fileEpoch, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    (_, testData_by1Nid, c, _) = initialize(nodeType, edgeType)
    
    print(commonFileName)
    printAllParams(fwEpoch)
    fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'time', 'aggrType'])
    d_global = c.get_d_global(True) ; d_group = c.get_d_group(True) ; d_sum = 0
    input_w_ks = [ None for _ in c.groups ] ; lr = hfl_param.LR_INITIAL
#     for t3 in range(int(hfl_param.MAX_EPOCH/(tau1*tau2))):
    t3 = 0
    while True:
        for t2 in range(tau2):
            w_k_byTime_byGid = []
            output_w_ks = []
            for k, g in enumerate(c.groups): # Group Aggregation
                w_k = input_w_ks[k]
                (w_k_byTime, _) = hfl_core.federated_train(w_k, g.get_Data_k_is(), lr, tau1, g.get_D_k_is())
                w_k_byTime_byGid.append(w_k_byTime)
                output_w_ks.append(w_k_byTime[-1])
            input_w_ks = output_w_ks
            w_byTime = hfl_core.federated_aggregate(w_k_byTime_byGid, c.get_D_ks()) # Global Aggregation
            for t1 in range(tau1):
                t = t3*tau1*tau2 + t2*tau1 + t1 + 1
                if t % tau1 == 0 and not(t % (tau1*tau2)) == 0:
                    d_sum += d_group
                    aggrType = 'Group'
                elif t % (tau1*tau2) == 0:
                    d_sum += d_global
                    aggrType = 'Global'
                else:
                    aggrType = ''
                (loss, accuracy) = hfl_core.evaluate(w_byTime[t1], testData_by1Nid[0])
                print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f\ttime=%.4f\taggrType=%s' % (t, loss, accuracy, d_sum, aggrType))
                fwEpoch.writerow([t, loss, accuracy, d_sum, aggrType])
                
                lr *= hfl_param.LR_DECAY_RATE
                if d_sum >= hfl_param.MAX_TIME: break;
            if d_sum >= hfl_param.MAX_TIME: break;
        if d_sum >= hfl_param.MAX_TIME: break;
        t3 += 1
        
        w = w_byTime[-1]
        input_w_ks = [ w for _ in c.groups ]
        
        # 실험 출력용 Delta 계산
#        (g_is__w, nid2_g_i__w) = hfl_core.federated_collect_gradients(w, c.get_nid2_Data_i())
#        g__w = np.average(g_is__w, axis=0, weights=c.get_D_is())
#        Delta = c.get_Delta(nid2_g_i__w, g__w)
#        print(Delta)
    print()
    fileEpoch.close()
    printTimedLogs(commonFileName)
    
def groupRandomly(numNodes, numGroups):
    numNodesPerGroup = int(numNodes / numGroups)
    z_rand = [ k for k in range(numGroups) for _ in range(numNodesPerGroup) ]
    np.random.shuffle(z_rand)
    return z_rand

def HiFlex(nodeType, edgeType, d_budget):
    commonFileName = TITLE_PARAMS + '_' + nodeType + '_' + edgeType + '_HiFlex_' + str(d_budget)
    fileEpoch = open('logs/' + commonFileName + '_epoch.csv', 'w', newline='', buffering=1)
    fileDelta = open('logs/' + commonFileName + '_Delta.csv', 'w', newline='', buffering=1)
    fwEpoch = csv.writer(fileEpoch, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    fwDelta = csv.writer(fileDelta, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    (_, testData_by1Nid, _, c) = initialize(nodeType, edgeType)
    
    # 그룹 멤버쉽 랜덤 초기화
#    z_rand = groupRandomly(len(c.get_N()), len(c.groups))
#    c.digest(z_rand)
    
    print(commonFileName)
    printAllParams(fwEpoch)
    fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'time', 'aggrType', 'numGroups', 'tau1', 'tau2'])
    t = 0 ; t_prev = 0 ; tau1 = 1 ; tau2 = 1 ; t3 = 0 ; d_sum = 0
    input_w_ks = [ None for _ in c.groups ] ; lr = hfl_param.LR_INITIAL
    d_group = c.get_d_group(True) ; d_global = c.get_d_global(True)
    while True:
        for t2 in range(tau2):
            w_is = []
            w_k_byTime_byGid = []
            output_w_ks = []
            for k, g in enumerate(c.groups): # Group Aggregation
                w_k = input_w_ks[k]
                (w_k_byTime, w_k_is) = hfl_core.federated_train(w_k, g.get_Data_k_is(), lr, tau1, g.get_D_k_is())
                w_is += w_k_is
                w_k_byTime_byGid.append(w_k_byTime)
                output_w_ks.append(w_k_byTime[-1])
            input_w_ks = output_w_ks
            w_byTime = hfl_core.federated_aggregate(w_k_byTime_byGid, c.get_D_ks()) # Global Aggregation
            for t1 in range(tau1):
                t += 1
                if (t-t_prev) % tau1 == 0 and not((t-t_prev) % (tau1*tau2)) == 0:
                    d_sum += d_group
                    aggrType = 'Group'
                elif (t-t_prev) % (tau1*tau2) == 0:
                    d_sum += d_global
                    aggrType = 'Global'
                else:
                    aggrType = ''
                (loss, accuracy) = hfl_core.evaluate(w_byTime[t1], testData_by1Nid[0])
                print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f\ttime=%.4f\taggrType=%s\tnumGroups=%d\ttau1=%d\ttau2=%d'
                      % (t, loss, accuracy, d_sum, aggrType, len(c.groups), tau1, tau2))
                fwEpoch.writerow([t, loss, accuracy, d_sum, aggrType, len(c.groups), tau1, tau2])
                
                lr *= hfl_param.LR_DECAY_RATE
#                 if t >= hfl_param.MAX_EPOCH: break
#             if t >= hfl_param.MAX_EPOCH: break
#         if t >= hfl_param.MAX_EPOCH: break
                if d_sum >= hfl_param.MAX_TIME: break;
            if d_sum >= hfl_param.MAX_TIME: break;
        if d_sum >= hfl_param.MAX_TIME: break;
        
        w = w_byTime[-1] # 출력을 위해 매 시간마다 수행했던 Global Aggregation 의 마지막 시간 값만 추출
        input_w_ks = [ w for _ in c.groups ]
        
        if t3 % hfl_param.HIFLEX_IID_GROUPING_INTERVAL == 0:
            # Gradient Estimation
            (g_is__w, nid2_g_i__w) = hfl_core.federated_collect_gradients(w, c.get_nid2_Data_i())
#             (g_is__w2, nid2_g_i__w2) = hfl_core.federated_collect_gradients2(w, c.get_nid2_Data_i())
#             for nid in nid2_g_i__w:
#                 print(hfl_core.np_modelEquals(g_is__w[nid], g_is__w2[nid]), hfl_core.np_modelEquals(nid2_g_i__w[nid], nid2_g_i__w2[nid]))
            g__w = np.average(g_is__w, axis=0, weights=c.get_D_is())

            # IID Grouping
            (c, tau1, tau2, d_group, d_global) = run_IID_Grouping(fwDelta, c, nid2_g_i__w, g__w, d_budget)
        t_prev = t # Mark Time
        t3 += 1
    print()
    fileEpoch.close()
    fileDelta.close()
    printTimedLogs(commonFileName)

def run_IID_Grouping(fwDelta, c, nid2_g_i__w, g__w, d_budget):
    print('IID Grouping Started')
    
    # 최적 초기화
    (c_star, Delta_star) = iterate_IID_Grouping(c, nid2_g_i__w, g__w)
    print('Initial Delta_star=%.4f' % (Delta_star))
    fwDelta.writerow(['time', 'cntSteady', 'Delta_star', 'Delta_cur'])
    fwDelta.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), 0, Delta_star, Delta_star])
    
    cntSteady = 0
    while cntSteady < hfl_param.HIFLEX_MAX_STEADY_STEPS:
        (c, Delta) = iterate_IID_Grouping(c, nid2_g_i__w, g__w)
        if Delta_star > Delta:
            (c_star, Delta_star) = (c, Delta)
            cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
        else:
            cntSteady += 1
        print('cntSteady=%d, Delta_star=%.4f, Delta_cur=%.4f' % (cntSteady, Delta_star, Delta))
        fwDelta.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), cntSteady, Delta_star, Delta])
        
    d_group = c.get_d_group(True)
    d_global = c.get_d_global(True)
    tau1 = 1
    if d_group < d_global:
        tau2 = max( int((d_budget + d_group - d_global) / d_group), 1 )
    else:
        tau2 = 2
    print('Final Delta_star=%.4f, tau1=%d, tau2=%d' % (Delta_star, tau1, tau2))
    
    print('IID Grouping Finished')
    return c_star, tau1, tau2, d_group, d_global

def iterate_IID_Grouping(c, nid2_g_i__w, g__w):
    Delta_cur = c.get_Delta(nid2_g_i__w, g__w)
    
    # Iteration 마다 탐색 인덱스 셔플
    idx_Nid1 = np.arange(len(c.get_N()))
    np.random.shuffle(idx_Nid1)
    idx_Nid1 = idx_Nid1[:hfl_param.HIFLEX_NUM_SAMPLE_NODE]
    
    idx_Nid2 = np.arange(len(c.get_N()))
    np.random.shuffle(idx_Nid2)
    idx_Nid2 = idx_Nid2[:hfl_param.HIFLEX_NUM_SAMPLE_NODE]
    
    z = c.z
    for i in idx_Nid1:
        for j in idx_Nid2:
            # 목적지 그룹이 이전 그룹과 같을 경우 무시
            if z[i] == z[j]: continue
                
            # 그룹 멤버쉽 변경 시도
            temp_k = z[i]
            z[i] = z[j]
            z[j] = temp_k
            
            nids_byGid = hfl_util.to_nids_byGid(z)
            numClassesPerGroup = np.mean([ len(np.unique(np.concatenate([c.data_byNid[nid]['y'] for nid in nids]))) for nids in nids_byGid ])
            if numClassesPerGroup != 10:
                # 데이터 분포가 IID 가 아니게 될 경우, 다시 원래대로 그룹 멤버쉽 초기화
                temp_k = z[i]
                z[i] = z[j]
                z[j] = temp_k
                continue
            
            # 한 그룹에 노드가 전부 없어지는 것은 있을 수 없는 경우
            if c.digest(z) == False: raise Exception(str(z))
            
            # 다음 후보에 대한 Delta 계산
            Delta_next = c.get_Delta(nid2_g_i__w, g__w)
            
            if Delta_cur > Delta_next:
                # 유리할 경우 이동
                Delta_cur = Delta_next
            else:
                # 유리하지 않을 경우, 다시 원래대로 그룹 멤버쉽 초기화 (다음 Iteration 에서 Digest)
                temp_k = z[i]
                z[i] = z[j]
                z[j] = temp_k
#         print(i, Delta_cur)
                
    # 마지막 멤버십 초기화가 발생했을 수도 있으므로, Cloud Digest 시도
    if c.digest(z) == False: raise Exception(str(z))
    return c, Delta_cur