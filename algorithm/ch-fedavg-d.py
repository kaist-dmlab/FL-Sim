import importlib
import numpy as np

from cloud.cloud import Cloud
import fl_data

ChFedavgPackagePath = 'algorithm.ch-fedavg'
ChFedavgModule = importlib.import_module(ChFedavgPackagePath)
ChFedavgAlgorithm = getattr(ChFedavgModule, 'Algorithm')

GROUPING_MAX_STEADY_STEPS = 1

NUM_NORM_CONST_SAMPLES = 50

class Algorithm(ChFedavgAlgorithm):
    
    def __init__(self, args):
        super().__init__(args)
        default_linkSpeed = args.linkSpeeds[0]
#         self.nid2delay = self.c.topology.profileDelay(1024, default_linkSpeed)
#         print('Group Delay', self.c.get_delay_group(self.nid2delay))
        print('Group HPP', self.c.get_hpp_group(True))
        
    def runGrouping(self, c, nid2_g_i__w, g__w, d_budget):
        print('Comm-IID Grouping Started')
        
        # 최적 초기화
        c_star = None
        cost_star = 1e9
        
        # 그룹 개수 바꿔가면서 탐색
        cntSteady = 0
        numGroups = 20 # numGroups = 1 일 때는, DELTA 가 0 이 되므로 다른 numGroups 경우와 Cost 비교 불가
        while cntSteady < GROUPING_MAX_STEADY_STEPS and numGroups <= 20:
            print('numGroups:', numGroups)
            c_cur = Cloud(c.topology, c.D_byNid, numGroups)
            z_rand = fl_data.groupRandomly(self.args.numNodes, numGroups)
            nids_byGid = fl_data.to_nids_byGid(z_rand)
            c_cur.digest(nids_byGid, nid2_g_i__w, g__w)
            print('Group HPP', c_cur.get_hpp_group(True))
        
            # Cost Normalization Constant 샘플링
            costs_comm = [] ; costs_iid = []
            for _ in range(NUM_NORM_CONST_SAMPLES):
#                 numGroups = np.random.randint(2, self.args.numNodes+1)
                c_sample = Cloud(c.topology, c.D_byNid, numGroups)
                z_rand = fl_data.groupRandomly(self.args.numNodes, numGroups)
                nids_byGid = fl_data.to_nids_byGid(z_rand)
                c_sample.digest(nids_byGid, nid2_g_i__w, g__w)
                costs_comm.append(c_sample.get_hpp_group(True))
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
        self.d_global = self.getCommTimeGlobal(c_star, self.model.size, default_linkSpeed)
        self.d_group = self.getCommTimeGroup(c_star, self.model.size, default_linkSpeed)
        tau1 = 1
        tau2 = 5
#         if self.d_group < self.d_global:
#             tau2 = max( int((d_budget + self.d_group - self.d_global) / self.d_group), 1 )
#         else:
#             tau2 = 2
        print('Final cost_star=%.3f, numGroups=%d, d_group=%.3f, d_global=%.3f, tau1=%d, tau2=%d' % \
              (cost_star, len(c_star.groups), self.d_group, self.d_global, tau1, tau2))
        print(c_star.get_hpp_group(True)/self.norm_const_comm, c_star.get_DELTA()/self.norm_const_iid)
        
        print('Comm-IID Grouping Finished')
        return c_star, tau1, tau2
    
    def getCost(self, c):
        if self.norm_const_comm == 0 or self.norm_const_iid == 0:
            raise Exception(str(self.norm_const_comm), str(self.norm_const_iid))
        return c.get_hpp_group(True)/self.norm_const_comm + c.get_DELTA()/self.norm_const_iid