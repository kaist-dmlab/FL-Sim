import importlib
import numpy as np

from cloud.cloud import Cloud
import fl_const
import fl_data

ChFedavgPackagePath = 'algorithm.ch-fedavg'
ChFedavgModule = importlib.import_module(ChFedavgPackagePath)
ChFedavgAlgorithm = getattr(ChFedavgModule, 'Algorithm')

GROUPING_MAX_STEADY_STEPS = 1

NUM_NORM_CONST_SAMPLES = 50

def removeOutliers(x, outlierConstant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList

class Algorithm(ChFedavgAlgorithm):
    
    def __init__(self, args):
        super().__init__(args)
        default_linkSpeed = args.linkSpeeds[0]
#         self.nid2delay = self.c.topology.profileDelay(1024, default_linkSpeed)
#         print('Group Delay', self.c.get_delay_group(self.nid2delay))
        
    def runGrouping(self, c, nid2_g_i__w, g__w):
        print('Comm-IID Grouping Started')
        
        # 최적 초기화
        c_star = None
        cost_star = 1e9
        
        # 그룹 개수 바꿔가면서 탐색
        cntSteady = 0
        numGroups = self.args.numGroups # numGroups = 1 일 때는, DELTA 가 0 이 되므로 다른 numGroups 경우와 Cost 비교 불가
        while cntSteady < GROUPING_MAX_STEADY_STEPS and numGroups <= self.args.numGroups:
            print('numGroups:', numGroups)
            c_cur = Cloud(c.topology, c.D_byNid, numGroups)
            z_rand = fl_data.groupRandomly(self.args.numNodes, numGroups)
            nids_byGid = fl_data.to_nids_byGid(z_rand)
            c_cur.digest(nids_byGid, nid2_g_i__w, g__w)
            
            # Cost Normalization Constant 샘플링
            costs_comm = [] ; costs_iid = []
            for _ in range(NUM_NORM_CONST_SAMPLES):
#                 numGroups = np.random.randint(2, self.args.numNodes+1)
                c_sample = Cloud(c.topology, c.D_byNid, numGroups)
                z_rand = fl_data.groupRandomly(self.args.numNodes, numGroups)
                nids_byGid = fl_data.to_nids_byGid(z_rand)
                c_sample.digest(nids_byGid, nid2_g_i__w, g__w)
                costs_comm.append(c_sample.get_hpp_group(False))
                costs_iid.append(c_sample.get_DELTA())
            self.norm_const_comm = np.mean(costs_comm)
            self.norm_const_iid = np.mean(costs_iid)
            print('Cost Norm Constant:', min(costs_comm), max(costs_comm), min(costs_iid), max(costs_iid))
            
            (c_cur, cost_cur) = self.runGroupingInternal(c_cur, nid2_g_i__w, g__w)
            if self.isGreaterThan(cost_star, cost_cur):
                (c_star, cost_star) = (c_cur.clone(nid2_g_i__w, g__w), cost_cur)
                cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
            else:
                cntSteady += 1
            numGroups += 1
        
        default_linkSpeed = self.args.linkSpeeds[0]
        d_group = self.getCommTimeGroup(c_star, self.model.size, default_linkSpeed)
        d_global = self.getCommTimeGlobal(c_star, self.model.size, default_linkSpeed)
        self.setDefaultCommDelay(d_group, d_global)
        
        print('Final cost_star=%.3f, numGroups=%d, d_group=%.3f, d_global=%.3f' % \
              (cost_star, len(c_star.groups), d_group, d_global))
        print(c_star.get_hpp_group(False)/self.norm_const_comm, c_star.get_DELTA()/self.norm_const_iid)
        
        print('Comm-IID Grouping Finished')
        return c_star
    
    def getCost(self, c):
        if self.norm_const_comm == 0 or self.norm_const_iid == 0:
            raise Exception(str(self.norm_const_comm), str(self.norm_const_iid))
        return c.get_hpp_group(False)/self.norm_const_comm + c.get_DELTA()/self.norm_const_iid
    
#     def getMedoidNid(self, g, nid2_g_i__w, g__w):
#         numPackets = fl_const.DEFAULT_PACKET_SIZE / fl_const.DEFAULT_PACKET_SIZE
#         N_k = g.get_N_k()
#         l = []
#         for i in N_k:
#             hpp_i = sum( g.topology.getDistance(i, i2) for i2 in N_k ) / numPackets
#             delta_i = np.linalg.norm(nid2_g_i__w[i] - g__w)
#             l.append( hpp_i/self.norm_const_comm + delta_i/self.norm_const_iid )
#         return N_k[ np.argmin(l) ]