import numpy as np
import os
import csv
from time import gmtime, strftime

from algorithm.abc import AbstractAlgorithm, abstractmethod

from cloud.cloud import Cloud
import fl_const
import fl_data

COST_CSV_POSTFIX = 'cost.csv'

WEIGHTING_MAX_STEADY_STEPS = 100
WEIGHTING_WEIGHT_DIFF = 10

GROUPING_INTERVAL = 10000
GROUPING_MAX_STEADY_STEPS = 1
GROUPING_ERROR_THRESHOLD = 0.01

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        tau1 = int(self.args.opaque1)
        tau2 = int(self.args.opaque2)
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName + '_' \
                + self.args.nodeType + self.args.edgeType + '_' + str(tau1) + '_' + str(tau2) + '_' + str(self.args.numGroups)
    
    def __init__(self, args):
        args.edgeCombineEnabled = False
        super().__init__(args)
        
        fileName = self.getFileName()
        self.fileCost = open(os.path.join(fl_const.LOG_DIR_PATH, fileName + '_' + COST_CSV_POSTFIX), 'w', newline='', buffering=1)
        self.fwCost = csv.writer(self.fileCost, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
    def __del__(self):
        self.fileCost.close()
        super().__del__()
    
    def getApprCommCostGroup(self, c):
        return c.get_max_hpp_group(self.args.edgeCombineEnabled) * 2
    
    def getApprCommCostGlobal(self, c):
        return c.get_hpp_global(self.args.edgeCombineEnabled) * 2
    
    def getCommTimeGroup(self, c, dataSize, linkSpeed):
        return c.get_d_group(self.args.edgeCombineEnabled, dataSize, linkSpeed) * 2
    
    def getCommTimeGlobal(self, c, dataSize, linkSpeed):
        return c.get_d_global(self.args.edgeCombineEnabled, dataSize, linkSpeed) * 2
    
    def run(self):
        self.fwCost.writerow(['time', 'cntSteady', 'cost_prev', 'cost_new'])
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'aggrType', 'numGroups', 'tau1', 'tau2'])
        
        lr = self.args.lrInitial
        
        self.epoch = 0 ; epoch_prev = 0
        tau1 = 1 ; tau2 = 1 ; t3 = 0 ; time = 0
        while True:
            input_w_ks = [ self.model.getParams() for _ in self.c.groups ]
            for t2 in range(tau2):
                w_is = []
                w_k_byTime_byGid = []
                output_w_ks = []
                for k, g in enumerate(self.c.groups): # Group Aggregation
                    w_k = input_w_ks[k]
                    (w_k_byTime, w_k_is) = self.model.federated_train(w_k, g.get_D_k_is(), lr, tau1, g.get_p_k_is())
                    w_is += w_k_is
                    w_k_byTime_byGid.append(w_k_byTime)
                    output_w_ks.append(w_k_byTime[-1])
                input_w_ks = output_w_ks
                w_byTime = self.model.federated_aggregate(w_k_byTime_byGid, self.c.get_p_ks()) # Global Aggregation
                for t1 in range(tau1):
                    self.epoch += 1
                    d_local, d_group, d_global = self.getDefaultDelay()
                    time += d_local
                    if (self.epoch-epoch_prev) % tau1 == 0 and not((self.epoch-epoch_prev) % (tau1*tau2)) == 0:
                        time += d_group
                        aggrType = 'Group'
                    elif (self.epoch-epoch_prev) % (tau1*tau2) == 0:
                        time += d_global
                        aggrType = 'Global'
                    else:
                        aggrType = ''
                    (loss, _, _, acc) = self.model.evaluate(w_byTime[t1])
                    print('epoch=%5d\ttime=%.3f\tloss=%.3f\taccuracy=%.3f\taggrType=%s\tnumGroups=%d\ttau1=%d\ttau2=%d'
                          % (self.epoch, time, loss, acc, aggrType, len(self.c.groups), tau1, tau2))
                    self.fwEpoch.writerow([self.epoch, loss, acc, aggrType, len(self.c.groups), tau1, tau2])
                    
                    lr *= self.args.lrDecayRate
                    if time >= self.args.maxTime: break;
                if time >= self.args.maxTime: break;
            if time >= self.args.maxTime: break;
                
            w = w_byTime[-1] # 출력을 위해 매 시간마다 수행했던 Global Aggregation 의 마지막 시간 값만 추출
            input_w_ks = [ w for _ in self.c.groups ]
            
            if t3 % GROUPING_INTERVAL == 0:
                # Gradient Estimation
                (g_is__w, nid2_g_i__w) = self.model.federated_collect_gradients(w, self.c.get_nid2_D_i())
                g__w = np.average(g_is__w, axis=0, weights=self.c.get_p_is())
                
                print([ np.linalg.norm(nid2_g_i__w[nid]) for nid in range(self.args.numNodes) ])
                
#                 D_is = self.c.get_D_is()
#                 p_is = [ len(np.unique(D_i['y'])) for D_i in D_is ]
#                 print(p_is)
#                 p_is = [ 1 for _ in range(len(self.c.get_N())) ]
#                 self.c.set_p_is(p_is)
                self.c.invalidate() # because all the groups need to be updated
                self.c.digest(self.c.nids_byGid, nid2_g_i__w, g__w)
        
#                 c = self.runIidWeighting(self.c, nid2_g_i__w, g__w)
                self.c = self.runGrouping(self.c, nid2_g_i__w, g__w)
                tau1 = int(self.args.opaque1)
                tau2 = int(self.args.opaque2)
            epoch_prev = self.epoch # Mark Time
            t3 += 1
            
    def runIidWeighting(self, c, nid2_g_i__w, g__w, mode=1):
        print('IID Weighting Started')
        
        c_star = c.clone(nid2_g_i__w, g__w)
        if mode == 1:
            delta_star = c.get_Delta(nid2_g_i__w, g__w)
        elif mode == 2:
            delta_star = c.get_delta()
        elif mode == 3:
            delta_star = c.get_DELTA()
        else:
            delta_star = c.get_emd()
        print('Initial delta_star=%.3f' % (delta_star))
        
        cntIdx = 0
        cntSteady = 0
        cntPrint = 0
        while cntSteady < WEIGHTING_MAX_STEADY_STEPS:
            curIdx = cntIdx % len(c.get_N())
            cntIdx += 1
            (c_star, delta_star, cntSteady, cntPrint) = self.improveWeightOnIdx(c, c_star, delta_star, nid2_g_i__w, g__w, curIdx, cntSteady, cntPrint, mode)
#             randIdx = np.random.randint(len(c.get_N()))
#             (c_star, delta_star, cntSteady, cntPrint) = self.improveWeightOnIdx(c, c_star, delta_star, nid2_g_i__w, g__w, randIdx, cntSteady, cntPrint, mode)

#         # Tweak weights
#         p_is = c.get_p_is()
#         for i in range(len(p_is)):
#             if np.random.choice([True, False]) == True:
#                 p_is[i] += WEIGHTING_WEIGHT_DIFF
#             else:
#                 if p_is[i] > WEIGHTING_WEIGHT_DIFF:
#                     p_is[i] -= WEIGHTING_WEIGHT_DIFF
#                 else:
#                     p_is[i] = 0
#         c.set_p_is(p_is)
#         c.digest(c.nids_byGid, nid2_g_i__w, g__w)
        
#         if mode == 1:
#             delta_cur = c.get_Delta(nid2_g_i__w, g__w)
#         elif mode == 2:
#             delta_cur = c.get_delta()
#         elif mode == 3:
#             delta_cur = c.get_DELTA()
#         else:
#             delta_cur = c.get_emd()
#         print('Tweaked delta_star=%.3f, delta_cur=%.3f' % (delta_star, delta_cur))
            
        print('IID Weighting Finished')
        return c_star
    
    def improveWeightOnIdx(self, c, c_star, delta_star, nid2_g_i__w, g__w, curIdx, cntSteady, cntPrint, mode):
        p_is = c.get_p_is()
        
        # 높은 Weight 로 변경 시도
        p_is[curIdx] += WEIGHTING_WEIGHT_DIFF
        c.set_p_is(p_is)
        c.digest(c.nids_byGid, nid2_g_i__w, g__w)
        
        if mode == 1:
            delta_up = c.get_Delta(nid2_g_i__w, g__w)
        elif mode == 2:
            delta_up = c.get_delta()
        elif mode == 3:
            delta_up = c.get_DELTA()
        else:
            delta_up = c.get_emd()
        if delta_star > delta_up:
            # 유리할 경우 변경
            (c_star, delta_star) = (c.clone(nid2_g_i__w, g__w), delta_up)
            cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
            cntPrint += 1
            if cntPrint % 100 == 0:
                print('delta_star=%.3f, delta_cur=%.3f' % (delta_star, delta_up))
                self.fwCost.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), cntSteady, delta_star, delta_up])
        else:
            # 유리하지 않을 경우,
            # 낮은 Weight 로 변경 시도 (2배: 원래대로 돌리는 것 포함)
            if p_is[curIdx] > 2 * WEIGHTING_WEIGHT_DIFF:
                p_is[curIdx] -= 2 * WEIGHTING_WEIGHT_DIFF
            else:
                p_is[curIdx] = 0
            c.set_p_is(p_is)
            c.digest(c.nids_byGid, nid2_g_i__w, g__w)
            
            if mode == 1:
                delta_down = c.get_Delta(nid2_g_i__w, g__w)
            elif mode == 2:
                delta_down = c.get_delta()
            elif mode == 3:
                delta_down = c.get_DELTA()
            else:
                delta_down = c.get_emd()
            if delta_star > delta_down:
                # 유리할 경우 변경
                (c_star, delta_star) = (c.clone(nid2_g_i__w, g__w), delta_down)
                cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
                cntPrint += 1
                if cntPrint % 100 == 0:
                    print('delta_star=%.3f, delta_cur=%.3f' % (delta_star, delta_down))
            else:
                # 유리하지 않을 경우, 다시 원래대로 초기화
                p_is[curIdx] += WEIGHTING_WEIGHT_DIFF
                c.set_p_is(p_is)
                c.digest(c.nids_byGid, nid2_g_i__w, g__w)
                cntSteady += 1
        return (c_star, delta_star, cntSteady, cntPrint)
    
    def runGrouping(self, c_edge, nid2_g_i__w, g__w):
        print('IID Grouping Started')
        
        c = Cloud(c_edge.topology, c_edge.D_byNid, self.args.numGroups)
        z_rand = fl_data.groupRandomly(self.args.numNodes, self.args.numGroups)
        nids_byGid = fl_data.to_nids_byGid(z_rand)
        c.digest(nids_byGid, nid2_g_i__w, g__w)
        
        c_star, cost_star = self.runKMedoidsGrouping(c, nid2_g_i__w, g__w)
        
        default_linkSpeed = self.args.linkSpeeds[0]
        d_group = self.getCommTimeGroup(c_star, self.model.size, default_linkSpeed)
        d_global = self.getCommTimeGlobal(c_star, self.model.size, default_linkSpeed)
        self.setDefaultCommDelay(d_group, d_global)
        
        print(c_star.ps_nid, len(c_star.get_N()), c_star.get_N())
        
        print('Final cost_star=%.3f, numGroups=%d, d_group=%.3f, d_global=%.3f' % \
              (cost_star, len(c_star.groups), d_group, d_global))
        
        print('IID Grouping Finished')
        return c_star
    
    def runKMedoidsGrouping(self, c, nid2_g_i__w, g__w):
        # k-Medoids with Voronoi Iteration
        # https://en.wikipedia.org/wiki/K-medoids
        
        # Initialize Medoids
        medoidNids = []
        cntEid = 0
        for gid in range(self.args.numGroups):
            nidsInEdge = c.topology.getNids(cntEid % self.args.numEdges)
            while True:
                randNid = np.random.choice(nidsInEdge)
                if randNid in medoidNids:
                    continue
                else:
                    medoidNids.append(randNid)
                    break
            cntEid += 1
#         medoidNids = np.random.choice(np.arange(self.args.numNodes), size=self.args.numGroups, replace=False)
        print('Initial medoidNids:', sorted(medoidNids))
        
        # Associate
        c_cur, cost_cur = self.associate(c, nid2_g_i__w, g__w, medoidNids)
        c_star, cost_star = c_cur, cost_cur
        print('Initial cost_star=%.3f' % (cost_star))
        self.fwCost.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), 0, cost_star, cost_star])
        
        # Iterate
        cntSteady = 0
        while cntSteady < GROUPING_MAX_STEADY_STEPS:
            cost_prev = cost_star
            
            # Determine medoids
            medoidNids = self.determineMedoidNids(c_cur, nid2_g_i__w, g__w)
            print('medoidNids:', sorted(medoidNids))
            
            # Associate nodes to medoids
            c_cur, cost_cur = self.associate(c_cur, nid2_g_i__w, g__w, medoidNids)
            if self.isGreaterThan(cost_star, cost_cur):
                c_star, cost_star = c_cur, cost_cur
                cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
            else:
                cntSteady += 1
            cost_new = cost_star
            self.fwCost.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), cntSteady, cost_prev, cost_new])
            print('cntSteady=%d, cost_prev=%.3f, cost_new=%.3f' % (cntSteady, cost_prev, cost_new))
        return c_star, cost_star
    
    def isGreaterThan(self, lhs, rhs):
        return (lhs - rhs) > (GROUPING_ERROR_THRESHOLD * rhs)
    
    def associate(self, c, nid2_g_i__w, g__w, medoidNids):
        c = c.clone(nid2_g_i__w, g__w)
        medoidNids_byGid = [ [nid] for nid in medoidNids ]
        if c.digest(medoidNids_byGid, nid2_g_i__w, g__w) == False: raise Exception(str(medoidNids_byGid))
        
        # Shuffle index every iteration
        randNids = np.arange(self.args.numNodes)
        np.random.shuffle(randNids)
        
        z = fl_data.to_z(self.args.numNodes, c.nids_byGid)
        for i in randNids:
            # Medoid 일 경우 무시
            if i in medoidNids: continue
                
            # Search for candidates with the same minimum cost
            costs = []
            for k, g in enumerate(c.groups):
                # 목적지 그룹이 이전 그룹과 같을 경우 무시
                if z[i] == k: continue
                    
                # 그룹 멤버쉽 변경 시도
                z[i] = k
                
                # Cloud Digest 시도
                nids_byGid = fl_data.to_nids_byGid(z)
                if c.digest(nids_byGid, nid2_g_i__w, g__w) == False: raise Exception(str(z), str(nids_byGid))
                    
                # 다음 후보에 대한 Cost 계산
                costs.append(self.getAssociateCost(c))
            costs = np.array(costs)
            min_ks = np.where(costs == costs.min())[0]
            min_k = np.random.choice(min_ks)
            z[i] = min_k # 다음 Iteration 에서 Digest
            cost_min = costs[min_k]
            print(min_k, costs)
            print('nid: %3d\tcurrent cost: %.3f' % (i, cost_min))
            
        # 마지막 멤버십 초기화가 발생했을 수도 있으므로, Cloud Digest 시도
        nids_byGid = fl_data.to_nids_byGid(z)
        if c.digest(nids_byGid, nid2_g_i__w, g__w) == False: raise Exception(str(z), str(nids_byGid))
            
        for k, g in enumerate(c.groups):
            p_k = c.get_p_ks()[k]
            print(g.ps_nid, g.get_N_k(), p_k*g.get_DELTA_k())
        print('DELTA: ', c.get_DELTA())
        return c, cost_min
    
    @abstractmethod
    def getAssociateCost(self, c):
        pass

    @abstractmethod
    def determineMedoidNids(self, c, nid2_g_i__w, g__w):
        pass