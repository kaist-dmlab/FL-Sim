import importlib
import numpy as np

from cloud.cloud import Cloud
import fl_data

FedavgIcBasePackagePath = 'algorithm.fedavg-ic-abc'
FedavgIcBaseModule = importlib.import_module(FedavgIcBasePackagePath)
FedavgIcBaseAlgorithm = getattr(FedavgIcBaseModule, 'Algorithm')

GROUPING_MAX_STEADY_STEPS = 1

NUM_NORM_CONST_SAMPLES = 50

# def removeOutliers(x, outlierConstant=1.5):
#     a = np.array(x)
#     upper_quartile = np.percentile(a, 75)
#     lower_quartile = np.percentile(a, 25)
#     IQR = (upper_quartile - lower_quartile) * outlierConstant
#     quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
#     resultList = []
#     for y in a.tolist():
#         if y >= quartileSet[0] and y <= quartileSet[1]:
#             resultList.append(y)
#     return resultList

class Algorithm(FedavgIcBaseAlgorithm):
    
#     def __init__(self, args):
#         super().__init__(args)
#         default_linkSpeed = args.linkSpeeds[0]
#         self.nid2delay = self.c.topology.profileDelay(1024, default_linkSpeed)
#         print('Group Delay', self.c.get_delay_group(self.nid2delay))
        
    def runGrouping(self, c_edge, nid2_g_i__w, g__w):
        print('Comm-IID Grouping Started')
        
        # 최적 초기화
        c_star = None
        cost_star = 1e9
        
        # 그룹 개수 바꿔가면서 탐색
        cntSteady = 0
        numGroups = self.args.numGroups # numGroups = 1 일 때는, DELTA 가 0 이 되므로 다른 numGroups 경우와 Cost 비교 불가
        while cntSteady < GROUPING_MAX_STEADY_STEPS and numGroups <= self.args.numGroups:
            print('numGroups:', numGroups)
            c_cur = Cloud(c_edge.topology, c_edge.D_byNid, numGroups)
            z_rand = fl_data.groupRandomly(self.args.numNodes, numGroups)
            nids_byGid = fl_data.to_nids_byGid(z_rand)
            c_cur.digest(nids_byGid, nid2_g_i__w, g__w)
            
            # Cost Normalization Constant 샘플링
            iid_costs = [] ; comm_costs = []
            iid_k_costs = [] ; comm_k_costs = []
            for _ in range(NUM_NORM_CONST_SAMPLES):
                c_sample = Cloud(c_edge.topology, c_edge.D_byNid, numGroups)
                z_rand = fl_data.groupRandomly(self.args.numNodes, numGroups)
                nids_byGid = fl_data.to_nids_byGid(z_rand)
                c_sample.digest(nids_byGid, nid2_g_i__w, g__w)
                
                iid_costs.append(c_sample.get_DELTA())
                comm_costs.append(c_sample.get_sum_hpp_group(self.args.edgeCombineEnabled))
                
                for g in c_sample.groups:
                    N_k = g.get_N_k()
                    costs = []
                    for nid1 in N_k:
                        iid_k_costs.append(np.linalg.norm(nid2_g_i__w[nid1] - g__w))
                        comm_k_costs.append(sum( c_sample.topology.getDistance(nid1, nid2) for nid2 in g.N_k ))
            self.norm_const_iid = np.mean(iid_costs)
            self.norm_const_comm = np.mean(comm_costs)
            self.norm_const_k_iid = np.mean(iid_k_costs)
            self.norm_const_k_comm = np.mean(comm_k_costs)
            print('Cost Norm Constant:', min(iid_costs), max(iid_costs), min(comm_costs), max(comm_costs), \
                  min(iid_k_costs), max(iid_k_costs)), min(comm_k_costs), max(comm_k_costs)
            
            c_cur, cost_cur = self.runKMedoidsGrouping(c_cur, nid2_g_i__w, g__w)
            if self.isGreaterThan(cost_star, cost_cur):
                c_star, cost_star = c_cur, cost_cur
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
        print(c_star.get_DELTA()/self.norm_const_iid, c_star.get_sum_hpp_group(self.args.edgeCombineEnabled)/self.norm_const_comm)
        
        print('Comm-IID Grouping Finished')
        return c_star
    
    def getAssociateCost(self, c):
        if self.norm_const_comm == 0 or self.norm_const_iid == 0:
            raise Exception(str(self.norm_const_iid), str(self.norm_const_comm))
        return c.get_DELTA()/self.norm_const_iid + c.get_sum_hpp_group(self.args.edgeCombineEnabled)/self.norm_const_comm
    
    def determineMedoidNids(self, c, nid2_g_i__w, g__w):
        medoidNids = []
        for g in c.groups:
            N_k = g.get_N_k()
            costs = []
            for nid1 in N_k:
                p_i = c.get_p_is()[nid1]
                iid_i = np.linalg.norm(nid2_g_i__w[nid1] - g__w)
                comm_i = sum( c.topology.getDistance(nid1, nid2) for nid2 in g.N_k )
                costs.append( p_i*iid_i/self.norm_const_k_iid + comm_i/self.norm_const_k_comm )
            medoidNids.append(N_k[ np.argmin(costs) ])
        return medoidNids