import numpy as np
import random

from scipy.stats import truncnorm
from collections import Counter

def flattenX(data_by1Nid):
    data_by1Nid[0]['x'] = np.array([ x_.flatten() for x_ in data_by1Nid[0]['x'] ], dtype=np.float32)
    
def sample(data_by1Nid, numSamples):
    return [ { 'x': data_by1Nid[0]['x'][:numSamples], 'y': data_by1Nid[0]['y'][:numSamples] } ]

def to_nids_byGid(z):
    gids = np.unique(z)
    for gid in range(len(gids)):
        if not(gid in gids): return None
    nids_byGid = [ [ nid for nid, gid in enumerate(z) if gid == gid_ ] for gid_ in gids ]
    return nids_byGid

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
#         weights = np.random.exponential(scale=1000, size=numNodes)
        weights = 0.8 * (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) + 0.1 # 모두 0보다 크게 하기 위해 0.1 ~ 0.9 범위로 정규화
        weightSum = sum(weights)
        ps = [ int(numSamples*weights[i]/weightSum) for i in range(numNodes) ]
        ps[-1] += numSamples - sum(ps) # int 내림으로 인해 부족한 부분 채우기
        if np.all(np.array(ps) > 0): break
    assert( numSamples == sum(ps) )
    return ps

def groupByNode_old(data_by1Nid, nodeType, numNodes):
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

def groupByEdge_old(data_by1Nid, nodeType, edgeType, numNodes, numEdges):
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
    data_byNid = groupByNode_old(data_by1Nid, nodeType, numNodes)
    data_byNid = np.array([ data_byNid[nid] for nids in nids_byEid for nid in nids ]) # Node 오름차순으로 데이터 정렬
    z = [ eid for eid in range(numEdges) for _ in range(numNodesPerEdge) ]
    return (data_byNid, z)
def sample_truncated_normal(mean=1, sd=1, low=1, upp=10, size=1):
    if (low == upp): # NUM_CLASS_NODE = NUM_CLASS_EDGE인 경우-->한 EDGE내 모든 NODE는 같은 CLASS를 가짐. 
        sampled = [int(np.rint(low))]*size
    elif (mean >= upp) : # NUM_CLASS_NODE = NUM_CLASS_EDGE인 경우-->한 EDGE내 모든 NODE는 같은 CLASS를 가짐. 
        sampled = [int(np.rint(upp))]*size
    elif (mean <= low):
        sampled = [int(np.rint(low))]*size
    else:
        if (upp-low)/2 < mean:
            upp = upp + 0.5
        elif (upp-low)/2 == mean:
            pass
        else:
            low = low - 0.5
        sampled = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(size)
        sampled = [int(a) for a in np.rint(sampled)]
    return sampled


def groupByNode(data_by1Nid, nodeType, numNodes):
    NUM_CLASS = len(np.unique(data_by1Nid[0]['y']))
    data_byNid, z = groupByEdge(data_by1Nid, nodeType, NUM_CLASS, numNodes, 1)
    return data_byNid

def groupByEdge(data_by1Nid, nodeType, edgeType, numNodes, numEdges):
    np.random.seed(1234)
    trainData_by1Nid = data_by1Nid
    NUM_CLASS = len(np.unique(trainData_by1Nid[0]['y']))

    NUM_CLASS_NODE = nodeType # average # of class in each node
    NUM_CLASS_EDGE = edgeType # average # of class in each edge
    assert(NUM_CLASS_NODE <= NUM_CLASS_EDGE)

    NUM_NODES = numNodes
    NUM_EDGES = numEdges
    NUM_NODES_PER_EDGE = int(NUM_NODES / NUM_EDGES)
    assert(NUM_NODES % NUM_EDGES == 0)    
    
    EDGE_CLASS_LIST_UNIQUE = np.zeros(NUM_CLASS-1).tolist()
    while(len(EDGE_CLASS_LIST_UNIQUE)!=NUM_CLASS): # check whether all class are sampled at least once
        EDGE_CLASS_LIST = []
        EDGE_NODE_CLASS_LIST = []
        
        NUM_CLASS_EDGE_SAMPLED_MEAN = -100
        
        while(np.abs(NUM_CLASS_EDGE_SAMPLED_MEAN - NUM_CLASS_EDGE)>0.1): # mean error should be lower than 0.1
            NUM_CLASS_EDGE_SAMPLED = sample_truncated_normal(mean=NUM_CLASS_EDGE, sd=1, low=1, upp=10, size=NUM_EDGES)
            NUM_CLASS_EDGE_SAMPLED_MEAN = np.mean(NUM_CLASS_EDGE_SAMPLED)
    
        for i in range(NUM_EDGES):
            class_list = np.random.choice(list(range(NUM_CLASS)), NUM_CLASS_EDGE_SAMPLED[i], replace=False).tolist()
            EDGE_CLASS_LIST.append(class_list)
            EDGE_NODE_CLASS_LIST.append([])
        EDGE_CLASS_LIST_UNIQUE = np.unique([item for sublist in EDGE_CLASS_LIST for item in sublist])

    NODE_CLASS_LIST_UNIQUE = np.zeros(NUM_CLASS-1).tolist()
    
    NODE_CLASS_NUM_SAMPLED_MEAN = -100
    SEARCH_LIMIT = 0
    min_error = 100000
    while((len(NODE_CLASS_LIST_UNIQUE) < NUM_CLASS) and (np.abs(NODE_CLASS_NUM_SAMPLED_MEAN-NUM_CLASS_NODE)>0.1)): # check whether all class are sampled at least once
        if SEARCH_LIMIT > 1000:
            break
        SEARCH_LIMIT += 1
        
        for i in range(NUM_EDGES):
            edge_class_list = EDGE_CLASS_LIST[i]
            EDGE_CLASS_LIST_REAL_FLAT = np.zeros(NUM_CLASS+1)
            while (len(EDGE_CLASS_LIST_REAL_FLAT) != len(EDGE_CLASS_LIST[i])): # check whether all nodes in a edge sample all class assigned to the edge
                EDGE_NODE_CLASS_LIST[i] = []
                for j in range(NUM_NODES_PER_EDGE):
                    num_class_in_node = sample_truncated_normal(mean=NUM_CLASS_NODE, sd=1, low=1, upp=len(edge_class_list),size=1)
                    class_in_node = np.random.choice(edge_class_list, num_class_in_node, replace=False).tolist()
                    EDGE_NODE_CLASS_LIST[i].append(class_in_node)
                EDGE_CLASS_LIST_REAL_FLAT = np.unique([item for sublist in EDGE_NODE_CLASS_LIST[i] for item in sublist])
            
        EDGE_NODE_CLASS_LIST_FLAT = [iitem for sublist in EDGE_NODE_CLASS_LIST for item in sublist for iitem in item]
        NODE_CLASS_LIST_UNIQUE = np.unique(EDGE_NODE_CLASS_LIST_FLAT)
        NODE_CLASS_NUM_SAMPLED_MEAN = np.mean([len(item) for sublist in EDGE_NODE_CLASS_LIST for item in sublist])
        
        min_error_new = np.abs(NODE_CLASS_NUM_SAMPLED_MEAN-NUM_CLASS_NODE)
        if min_error_new < min_error:
            EDGE_NODE_CLASS_LIST_SAVED = EDGE_NODE_CLASS_LIST
    
    EDGE_NODE_CLASS_LIST = EDGE_NODE_CLASS_LIST_SAVED
    EDGE_NODE_CLASS_LIST_FLAT = [iitem for sublist in EDGE_NODE_CLASS_LIST for item in sublist for iitem in item]
    EDGE_NODE_CLASS_LIST_FLAT_2 = [item for sublist in EDGE_NODE_CLASS_LIST for item in sublist]
    Class_Counter = Counter(EDGE_NODE_CLASS_LIST_FLAT) # number of sampled class in all nodes

    DATA_BY_CLASS = groupByClass(trainData_by1Nid)
    for i in range(NUM_CLASS): # ensure data type
        DATA_BY_CLASS[i]['x'] = DATA_BY_CLASS[i]['x'].astype(np.float32)
        DATA_BY_CLASS[i]['y'] = DATA_BY_CLASS[i]['y'].astype(np.int32)
    
    
    NUM_DATA_PER_CLASS = [len(x['x']) for x in DATA_BY_CLASS]

    NUM_DATA_PER_SAMPLED_CLASS = []
    for i in range(NUM_CLASS):
        NUM_DATA_PER_SAMPLED_CLASS.append(DATA_BY_CLASS[i]['x'].shape[0]/Class_Counter[i])

    trainData_byNid = []
    train_z = []
    
    for edge_ind in range(NUM_EDGES):
        for node_class_list in EDGE_NODE_CLASS_LIST[edge_ind]:
            initial_counter = 0
            train_z.append(edge_ind)
            for i in node_class_list: # i-th node
                if initial_counter == 0: # make initial x,y dictionary for each node
                    num_sampled = sample_truncated_normal(mean=NUM_DATA_PER_SAMPLED_CLASS[i], sd = 1, low=1, upp=2*NUM_DATA_PER_SAMPLED_CLASS[i], size=1)[0] # number of datapoint for a class in a node
                    if NUM_DATA_PER_CLASS[i] > num_sampled:
                        indice_sampled = np.random.choice(range(NUM_DATA_PER_CLASS[i]), size=num_sampled, replace=False).tolist()
                        trainData_byNid.append({'x':DATA_BY_CLASS[i]['x'][indice_sampled] , 'y':DATA_BY_CLASS[i]['y'][indice_sampled]})
                        DATA_BY_CLASS[i]['x'] = np.delete(DATA_BY_CLASS[i]['x'], indice_sampled,0)
                        DATA_BY_CLASS[i]['y'] = np.delete(DATA_BY_CLASS[i]['y'], indice_sampled,0)
                        NUM_DATA_PER_CLASS[i] = NUM_DATA_PER_CLASS[i] - num_sampled
                        initial_counter += 1
                    else:
                        num_sampled = NUM_DATA_PER_CLASS[i]
                        indice_sampled = np.random.choice(range(NUM_DATA_PER_CLASS[i]), size=num_sampled, replace=False).tolist()
                        trainData_byNid.append({'x':DATA_BY_CLASS[i]['x'][indice_sampled] , 'y':DATA_BY_CLASS[i]['y'][indice_sampled]})
                        DATA_BY_CLASS[i]['x'] = np.delete(DATA_BY_CLASS[i]['x'], indice_sampled,0)
                        DATA_BY_CLASS[i]['y'] = np.delete(DATA_BY_CLASS[i]['y'], indice_sampled,0)
                        NUM_DATA_PER_CLASS[i] = 0
                        initial_counter += 1
                else: # add next datapoint from other class to x,y dictionary for each node
                    num_sampled = sample_truncated_normal(mean=NUM_DATA_PER_SAMPLED_CLASS[i], sd = 1, low=1, upp=2*NUM_DATA_PER_SAMPLED_CLASS[i], size=1)[0] # number of datapoint for a class in a node
                    if NUM_DATA_PER_CLASS[i] > num_sampled:
                        indice_sampled = np.random.choice(range(NUM_DATA_PER_CLASS[i]), size=num_sampled, replace=False).tolist()
                        trainData_byNid[-1]['x'] = np.concatenate([trainData_byNid[-1]['x'],DATA_BY_CLASS[i]['x'][indice_sampled]])
                        trainData_byNid[-1]['y'] = np.concatenate([trainData_byNid[-1]['y'],DATA_BY_CLASS[i]['y'][indice_sampled]])
                        DATA_BY_CLASS[i]['x'] = np.delete(DATA_BY_CLASS[i]['x'], indice_sampled,0)
                        DATA_BY_CLASS[i]['y'] = np.delete(DATA_BY_CLASS[i]['y'], indice_sampled,0)
                        NUM_DATA_PER_CLASS[i] = NUM_DATA_PER_CLASS[i] - num_sampled
                    else:
                        num_sampled = NUM_DATA_PER_CLASS[i]
                        indice_sampled = np.random.choice(range(NUM_DATA_PER_CLASS[i]), size=num_sampled, replace=False).tolist()
                        trainData_byNid[-1]['x'] = np.concatenate([trainData_byNid[-1]['x'],DATA_BY_CLASS[i]['x'][indice_sampled]])
                        trainData_byNid[-1]['y'] = np.concatenate([trainData_byNid[-1]['y'],DATA_BY_CLASS[i]['y'][indice_sampled]])
                        DATA_BY_CLASS[i]['x'] = np.delete(DATA_BY_CLASS[i]['x'], indice_sampled,0)
                        DATA_BY_CLASS[i]['y'] = np.delete(DATA_BY_CLASS[i]['y'], indice_sampled,0)
                        NUM_DATA_PER_CLASS[i] = 0

    nids_byGid = to_nids_byGid(train_z)
    trainData_byNid_NODE_CLASS_LIST = [np.unique(trainData_byNid[nid]['y']).tolist() for nids in nids_byGid for nid in nids]
    for i in range(NUM_CLASS):
        if NUM_DATA_PER_CLASS[i] != 0:
            node_index = 0
            node_cand = []
            for j in trainData_byNid_NODE_CLASS_LIST:
                if i in j:
                    node_cand.append(node_index)
                node_index += 1
            if NUM_DATA_PER_CLASS[i] > len(node_cand):
                node_samp = np.random.choice(node_cand, size = NUM_DATA_PER_CLASS[i], replace=True).tolist()
            else:
                node_samp = np.random.choice(node_cand, size = NUM_DATA_PER_CLASS[i], replace=False).tolist()
            for k in node_samp:
                trainData_byNid[k]['x'] = np.concatenate([trainData_byNid[k]['x'],DATA_BY_CLASS[i]['x'][[0]]])
                trainData_byNid[k]['y'] = np.concatenate([trainData_byNid[k]['y'],DATA_BY_CLASS[i]['y'][[0]]])
                DATA_BY_CLASS[i]['x'] = np.delete(DATA_BY_CLASS[i]['x'], 0,0)
                DATA_BY_CLASS[i]['y'] = np.delete(DATA_BY_CLASS[i]['y'], 0,0)
                NUM_DATA_PER_CLASS[i] = NUM_DATA_PER_CLASS[i] - 1

    for i in range(NUM_CLASS):
        assert(NUM_DATA_PER_CLASS[i] == 0)
        assert(len(DATA_BY_CLASS[i]['x']) == 0)
        assert(len(DATA_BY_CLASS[i]['y']) == 0)
    
    datanum = 0
    for node in trainData_byNid:
        datanum += len(node['x'])
    assert(datanum == len(data_by1Nid[0]['x']))
    
    return (trainData_byNid, train_z)