import numpy as np
import random

def preprocess(modelName, dataName, data, flatten=True):
    x = np.array([ x.flatten() / 255.0 if flatten else x / 255.0 for x in data[0] ], dtype=np.float32)
    dataY = data[1].flatten() # cifar10 의 경우 flatten 필요
    if modelName == 'svm':
        if dataName == 'cifar10':
            # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
            vehicleClasses = [0, 1, 8, 9]
            y = np.array([ -1 if y in vehicleClasses else 1 for y in dataY ], dtype=np.int32)
        else:
            raise Exception(dataName)
    else:
        y = np.array(dataY, dtype=np.int32)
    return np.array([ { 'x': x, 'y': y } ])

def sample(data_by1Nid, numSamples):
    return [ { 'x': data_by1Nid[0]['x'][:numSamples], 'y': data_by1Nid[0]['y'][:numSamples] } ]

def groupByClass(data_by1Nid):
    data_byClass = []
    for c in np.unique(data_by1Nid[0]['y']):
        idxExamplesForC = [ i for i, label in enumerate(data_by1Nid[0]['y']) if label == c ]
        data_byClass.append({ 'x': data_by1Nid[0]['x'][idxExamplesForC], 'y': data_by1Nid[0]['y'][idxExamplesForC] })
    return data_byClass

def partitionSumRandomly(numSamples, numNodes):
    # 0을 방지하기 위해 Half Random, Half Uniform
    np.random.seed(1234)
    mu, sigma = 0, 1 # 표준 정규분포로 Sample 개수 가중치 생성
    while True:
        weights = np.random.normal(mu, sigma, numNodes)
        weights = 0.8 * (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) + 0.1 # 모두 0보다 크게 하기 위해 0.1 ~ 0.9 범위로 정규화
        weightSum = sum(weights)
        ps = [ int(numSamples*weights[i]/weightSum) for i in range(numNodes) ]
        ps[-1] += numSamples - sum(ps) # int 내림으로 인해 부족한 부분 채우기
        if np.all(np.array(ps) > 0): break
    assert( numSamples == sum(ps) )
    return ps

def groupByNode(data_by1Nid, nodeType, numNodes):
    numClasses = len(np.unique(data_by1Nid[0]['y']))
    if not(numNodes % numClasses == 0): raise Exception(str(numNodes) + ' ' + str(numClasses))
    data_byClass = groupByClass(data_by1Nid)
    data_byNid = []
    if nodeType == 'o':
        numNodesPerClass = int(numNodes / numClasses)
        for c in range(numClasses):
            curNumSamples = len(data_byClass[c]['y'])
            ps = partitionSumRandomly(curNumSamples, numNodesPerClass)
            idxStart = 0 ; idxEnd = 0
            for p in ps:
                idxEnd += p
                data_byNid.append( { 'x': np.array(data_byClass[c]['x'][idxStart:idxEnd], dtype=np.float32),
                                     'y': np.array(data_byClass[c]['y'][idxStart:idxEnd], dtype=np.int32) } )
                idxStart += p
    elif nodeType == 'f':
        numClassSet = 5
        numNodesPerClassSet = int(numNodes / numClassSet)
        numClassesPerClassSet = int(numClasses / numClassSet)
        for cs in range(numClassSet):
            curData_byNid = []
            for c_ in range(numClassesPerClassSet):
                c = cs * numClassesPerClassSet + c_
                curNumSamples = len(data_byClass[c]['y'])
                ps = partitionSumRandomly(curNumSamples, numNodesPerClassSet)
                idxStart = 0 ; idxEnd = 0
                for i_ in range(numNodesPerClassSet):
                    idxEnd += ps[i_]
                    if len(curData_byNid) < numNodesPerClassSet:
                        curData_byNid.append( { 'x': np.array(data_byClass[c]['x'][idxStart:idxEnd], dtype=np.float32),
                                                'y': np.array(data_byClass[c]['y'][idxStart:idxEnd], dtype=np.int32) } )
                    else:
                        curData_byNid[i_]['x'] = np.append(curData_byNid[i_]['x'], data_byClass[c]['x'][idxStart:idxEnd], axis=0)
                        curData_byNid[i_]['y'] = np.append(curData_byNid[i_]['y'], data_byClass[c]['y'][idxStart:idxEnd])
                    idxStart += ps[i_]
            data_byNid += curData_byNid
    elif nodeType == 'h':
        numClassSet = 2
        numNodesPerClassSet = int(numNodes / numClassSet)
        numClassesPerClassSet = int(numClasses / numClassSet)
        for cs in range(numClassSet):
            curData_byNid = []
            for c_ in range(numClassesPerClassSet):
                c = cs * numClassesPerClassSet + c_
                curNumSamples = len(data_byClass[c]['y'])
                ps = partitionSumRandomly(curNumSamples, numNodesPerClassSet)
                idxStart = 0 ; idxEnd = 0
                for i_ in range(numNodesPerClassSet):
                    idxEnd += ps[i_]
                    if len(curData_byNid) < numNodesPerClassSet:
                        curData_byNid.append( { 'x': np.array(data_byClass[c]['x'][idxStart:idxEnd], dtype=np.float32),
                                                'y': np.array(data_byClass[c]['y'][idxStart:idxEnd], dtype=np.int32) } )
                    else:
                        curData_byNid[i_]['x'] = np.append(curData_byNid[i_]['x'], data_byClass[c]['x'][idxStart:idxEnd], axis=0)
                        curData_byNid[i_]['y'] = np.append(curData_byNid[i_]['y'], data_byClass[c]['y'][idxStart:idxEnd])
                    idxStart += ps[i_]
            data_byNid += curData_byNid
    elif nodeType == 'a':
        for c in range(numClasses):
            curNumSamples = len(data_byClass[c]['y'])
            ps = partitionSumRandomly(curNumSamples, numNodes)
            idxStart = 0 ; idxEnd = 0
            for i_ in range(numNodes):
                idxEnd += ps[i_]
                if len(data_byNid) < numNodes:
                    data_byNid.append( { 'x': np.array(data_byClass[c]['x'][idxStart:idxEnd], dtype=np.float32),
                                         'y': np.array(data_byClass[c]['y'][idxStart:idxEnd], dtype=np.int32) } )
                else:
                    data_byNid[i_]['x'] = np.append(data_byNid[i_]['x'], data_byClass[c]['x'][idxStart:idxEnd], axis=0)
                    data_byNid[i_]['y'] = np.append(data_byNid[i_]['y'], data_byClass[c]['y'][idxStart:idxEnd])
                idxStart += ps[i_]
    else:
        raise Exception(nodeType)
    return np.array(data_byNid)

def groupByEdge(modelName, dataName, data, nodeType, edgeType, numNodes, numEdges, flatten=True):
    data_by1Nid = preprocess(modelName, dataName, data, flatten)
    numNodesPerEdge = int(numNodes / numEdges)
    numClasses = len(np.unique(data_by1Nid[0]['y']))
    if not(numEdges % numClasses == 0): raise Exception(str(numEdges) + ' ' + str(numClasses))
    if not(numNodesPerEdge % numClasses == 0): raise Exception(str(numNodesPerEdge) + ' ' + str(numClasses))
    numNodesPerClass = int(numNodes / numClasses)
    nids_byEid = []
    if (nodeType == 'o' and edgeType == 'o') \
        or (nodeType == 'f' and edgeType == 'f') \
        or (nodeType == 'h' and edgeType == 'h') \
        or (nodeType == 'a' and edgeType == 'a'):
        nids_byEid = [ [ k * numNodesPerEdge + i_ for i_ in range(numNodesPerEdge) ] for k in range(numEdges) ]
    elif nodeType == 'o' and edgeType == 'f':
        numClassSet = 5
        for k in range(numClassSet):
            for j in range(numNodesPerClass):
                nids_byEid.append([ k * int(numNodes / numClassSet) + j + i * numNodesPerClass for i in range(int(numClasses / numClassSet)) ])
        nids_byEid = np.array(nids_byEid).reshape((numEdges, numNodesPerEdge)).tolist()
    elif nodeType == 'o' and edgeType == 'h':
        numClassSet = 2
        for k in range(numClassSet):
            for j in range(numNodesPerClass):
                nids_byEid.append([ k * int(numNodes / numClassSet) + j + i * numNodesPerClass for i in range(int(numClasses / numClassSet)) ])
        nids_byEid = np.array(nids_byEid).reshape((numEdges, numNodesPerEdge)).tolist()
    elif nodeType == 'f' and edgeType == 'h':
        numClassSet = 2
        for k in range(numClassSet):
            for j in range(numNodesPerClass):
                nids_byEid.append([ k * int(numNodes / numClassSet) + j + i * numNodesPerClass for i in range(int(numClasses / numClassSet)) ])
        nids_byEid = np.array(nids_byEid).reshape((numEdges, numNodesPerEdge)).tolist()
    elif (nodeType == 'o' and edgeType == 'a') \
        or (nodeType == 'f' and edgeType == 'a') \
        or (nodeType == 'h' and edgeType == 'a'):
        nids_byEid = [ [ j + i * numEdges for i in range(numNodesPerEdge) ] for j in range(numEdges) ]
    else:
        raise Exception(nodeType, edgeType)
    data_byNid = groupByNode(data_by1Nid, nodeType, numNodes)
    data_byNid = np.array([ data_byNid[nid] for nids in nids_byEid for nid in nids ]) # Node 오름차순으로 데이터 정렬
    z = [ eid for eid in range(numEdges) for _ in range(numNodesPerEdge) ]
    return (data_byNid, z)