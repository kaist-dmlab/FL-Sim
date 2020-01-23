import importlib
import numpy as np

from algorithm.abc import groupRandomly

from cloud.cloud import Cloud

ChFedavgPackagePath = 'algorithm.ch-fedavg'
ChFedavgModule = importlib.import_module(ChFedavgPackagePath)
ChFedavgAlgorithm = getattr(ChFedavgModule, 'Algorithm')

GROUPING_MAX_STEADY_STEPS = 1

NUM_NORM_CONST_SAMPLES = 50

class Algorithm(ChFedavgAlgorithm):
    
    def run_Grouping(self, c, nid2_g_i__w, g__w, d_budget):
        print('Comm-IID Grouping Started')
        
        # 최적 초기화
        c_star = None
        cost_star = 1e9
        
        # 그룹 개수 바꿔가면서 탐색
        cntSteady = 0 ; numGroups = 2 # numGroups = 1 일 때는, DELTA 가 0 이 되므로 다른 numGroups 경우와 Cost 비교 불가
        while cntSteady < GROUPING_MAX_STEADY_STEPS and numGroups <= self.args.numNodes:
            print('numGroups:', numGroups)
            c_cur = Cloud(c.topology, c.D_byNid, numGroups)
            z_rand = groupRandomly(self.args.numNodes, numGroups)
            c_cur.digest(z_rand, nid2_g_i__w, g__w)
            
            # Cost Normalization Constant 샘플링
            c_sample = c_cur.clone(nid2_g_i__w, g__w)
            costs_comm = [] ; costs_iid = []
            for _ in range(NUM_NORM_CONST_SAMPLES):
                z_rand = groupRandomly(self.args.numNodes, numGroups)
                c_sample.digest(z_rand, nid2_g_i__w, g__w)
                costs_comm.append(c_sample.get_hpp_group(True))
                costs_iid.append(c_sample.get_DELTA())
            self.norm_const_comm = np.mean(costs_comm)
            self.norm_const_iid = np.mean(costs_iid)
            print('Cost Norm Constant:', min(costs_comm), max(costs_comm), min(costs_iid), max(costs_iid))
#             print('norm_const_comm:', self.norm_const_comm, ', norm_const_iid:', self.norm_const_iid)
            
            (c_cur, cost_cur) = self.run_GroupingInternal(c_cur, nid2_g_i__w, g__w)
            if self.isGreaterThan(cost_star, cost_cur):
                (c_star, cost_star) = (c_cur.clone(nid2_g_i__w, g__w), cost_cur)
                cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
            else:
                cntSteady += 1
            numGroups += 1
        
        d_global = c_star.get_d_global(True)
        d_group = c_star.get_d_group(True)
        tau1 = 1
        if d_group < d_global:
            tau2 = max( int((d_budget + d_group - d_global) / d_group), 1 )
        else:
            tau2 = 2
        print('Final cost_star=%.4f, tau1=%d, tau2=%d' % (cost_star, tau1, tau2))
        
        print('Comm-IID Grouping Finished')
        return c_star, tau1, tau2
    
    def getCost(self, c):
        if self.norm_const_comm == 0 or self.norm_const_iid == 0:
            raise Exception(str(self.norm_const_comm), str(self.norm_const_iid))
        return c.get_hpp_group(True)/self.norm_const_comm + c.get_DELTA()/self.norm_const_iid