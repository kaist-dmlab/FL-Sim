import numpy as np
import os
import csv
import random
from time import gmtime, strftime

from algorithm.abc import AbstractAlgorithm
import fl_data

LOG_DIR_NAME = 'log'
COST_CSV_POSTFIX = 'cost.csv'

WEIGHTING_MAX_STEADY_STEPS = 100
WEIGHTING_WEIGHT_DIFF = 10

GROUPING_INTERVAL = 10000
GROUPING_MAX_STEADY_STEPS = 1
GROUPING_NUM_SAMPLE_NODE = 100

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        d_budget = self.args.opaque1
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName + '_' \
                + self.args.nodeType + self.args.edgeType + '_' + str(d_budget)
    
    def getApprCommCostGroup(self):
        return self.c.get_hpp_group(True) * 2
    
    def getApprCommCostGlobal(self):
        return self.c.get_hpp_global(True) * 3
    
    def getCommTimeGroup(self, linkSpeed):
        return self.c.get_d_group(True, linkSpeed) * 2
    
    def getCommTimeGlobal(self, linkSpeed):
        return self.c.get_d_global(True, linkSpeed) * 3
    
    def __init__(self, args):
        super().__init__(args, randomEnabled=True)
        
        fileName = self.getFileName()
        self.fileCost = open(os.path.join(LOG_DIR_NAME, fileName + '_' + COST_CSV_POSTFIX), 'w', newline='', buffering=1)
        self.fwCost = csv.writer(self.fileCost, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
    def __del__(self):
        self.fileCost.close()
        super().__del__()
        
    def run(self):
        self.fwCost.writerow(['time', 'cntSteady', 'cost_star', 'cost_cur'])
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'aggrType', 'numGroups', 'tau1', 'tau2'])
        
        lr = self.args.lrInitial
        input_w_ks = [ self.model.getParams() for _ in self.c.groups ]
        d_budget = self.args.opaque1
        
        self.epoch = 0 ; epoch_prev = 0
        tau1 = 1 ; tau2 = 1 ; t3 = 0 ; time = 0
        while True:
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
                    time += self.d_local
                    if (self.epoch-epoch_prev) % tau1 == 0 and not((self.epoch-epoch_prev) % (tau1*tau2)) == 0:
                        time += self.d_group
                        aggrType = 'Group'
                    elif (self.epoch-epoch_prev) % (tau1*tau2) == 0:
                        time += self.d_global
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
    #             (g_is__w2, nid2_g_i__w2) = self.model.federated_collect_gradients2(w, self.c.get_nid2_D_i())
    #             for nid in nid2_g_i__w:
    #                 print(self.model.np_modelEquals(g_is__w[nid], g_is__w2[nid]), self.model.np_modelEquals(nid2_g_i__w[nid], nid2_g_i__w2[nid]))
                g__w = np.average(g_is__w, axis=0, weights=self.c.get_p_is())
        
#                 D_is = self.c.get_D_is()
#                 p_is = [ len(np.unique(D_i['y'])) for D_i in D_is ]
#                 print(p_is)
#                 p_is = [ 1 for _ in range(len(self.c.get_N())) ]
#                 self.c.set_p_is(p_is)
#                 self.c.digest(self.c.z)

#                 c = self.run_IID_Weighting(self.c, nid2_g_i__w, g__w)
                (self.c, tau1, tau2) = self.run_Grouping(self.c, nid2_g_i__w, g__w, d_budget)
            epoch_prev = self.epoch # Mark Time
            t3 += 1
            
    def run_IID_Weighting(self, c, nid2_g_i__w, g__w, mode=1):
        print('IID Weighting Started')
        
        c_star = c.clone()
        if mode == 1:
            delta_star = c.get_Delta(nid2_g_i__w, g__w)
        elif mode == 2:
            delta_star = c.get_delta(nid2_g_i__w)
        elif mode == 3:
            delta_star = c.get_DELTA(nid2_g_i__w, g__w)
        else:
            delta_star = c.get_emd()
        print('Initial delta_star=%.4f' % (delta_star))
        
        cntIdx = 0
        cntSteady = 0
        cntPrint = 0
        while cntSteady < WEIGHTING_MAX_STEADY_STEPS:
            curIdx = cntIdx % len(c.get_N())
            cntIdx += 1
            (c_star, delta_star, cntSteady, cntPrint) = self.improveWeightOnIdx(c, c_star, delta_star, nid2_g_i__w, g__w, curIdx, cntSteady, cntPrint, mode)
#             randIdx = random.randint(0, len(c.get_N())-1)
#             (c_star, delta_star, cntSteady, cntPrint) = self.improveWeightOnIdx(c, c_star, delta_star, nid2_g_i__w, g__w, randIdx, cntSteady, cntPrint, mode)

#         # Tweak weights
#         p_is = c.get_p_is()
#         for i in range(len(p_is)):
#             if random.choice([True, False]) == True:
#                 p_is[i] += WEIGHTING_WEIGHT_DIFF
#             else:
#                 if p_is[i] > WEIGHTING_WEIGHT_DIFF:
#                     p_is[i] -= WEIGHTING_WEIGHT_DIFF
#                 else:
#                     p_is[i] = 0
#         c.set_p_is(p_is)
#         c.digest(c.z)
        
#         if mode == 1:
#             delta_cur = c.get_Delta(nid2_g_i__w, g__w)
#         elif mode == 2:
#             delta_cur = c.get_delta(nid2_g_i__w)
#         elif mode == 3:
#             delta_cur = c.get_DELTA(nid2_g_i__w, g__w)
#         else:
#             delta_cur = c.get_emd()
#         print('Tweaked delta_star=%.4f, delta_cur=%.4f' % (delta_star, delta_cur))
            
        print('IID Weighting Finished')
        return c_star
    
    def improveWeightOnIdx(self, c, c_star, delta_star, nid2_g_i__w, g__w, curIdx, cntSteady, cntPrint, mode):
        p_is = c.get_p_is()
        
        # 높은 Weight 로 변경 시도
        p_is[curIdx] += WEIGHTING_WEIGHT_DIFF
        c.set_p_is(p_is)
        c.digest(c.z)
        
        if mode == 1:
            delta_up = c.get_Delta(nid2_g_i__w, g__w)
        elif mode == 2:
            delta_up = c.get_delta(nid2_g_i__w)
        elif mode == 3:
            delta_up = c.get_DELTA(nid2_g_i__w, g__w)
        else:
            delta_up = c.get_emd()
        if delta_star > delta_up:
            # 유리할 경우 변경
            (c_star, delta_star) = (c.clone(), delta_up)
            cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
            cntPrint += 1
            if cntPrint % 100 == 0:
                print('delta_star=%.4f, delta_cur=%.4f' % (delta_star, delta_up))
                self.fwCost.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), cntSteady, delta_star, delta_up])
        else:
            # 유리하지 않을 경우,
            # 낮은 Weight 로 변경 시도 (2배: 원래대로 돌리는 것 포함)
            if p_is[curIdx] > 2 * WEIGHTING_WEIGHT_DIFF:
                p_is[curIdx] -= 2 * WEIGHTING_WEIGHT_DIFF
            else:
                p_is[curIdx] = 0
            c.set_p_is(p_is)
            c.digest(c.z)
            
            if mode == 1:
                delta_down = c.get_Delta(nid2_g_i__w, g__w)
            elif mode == 2:
                delta_down = c.get_delta(nid2_g_i__w)
            elif mode == 3:
                delta_down = c.get_DELTA(nid2_g_i__w, g__w)
            else:
                delta_down = c.get_emd()
            if delta_star > delta_down:
                # 유리할 경우 변경
                (c_star, delta_star) = (c.clone(), delta_down)
                cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
                cntPrint += 1
                if cntPrint % 100 == 0:
                    print('delta_star=%.4f, delta_cur=%.4f' % (delta_star, delta_down))
                    self.fwCost.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), cntSteady, delta_star, delta_down])
            else:
                # 유리하지 않을 경우, 다시 원래대로 초기화
                p_is[curIdx] += WEIGHTING_WEIGHT_DIFF
                c.set_p_is(p_is)
                c.digest(c.z)
                cntSteady += 1
        return (c_star, delta_star, cntSteady, cntPrint)
    
    def run_Grouping(self, c, nid2_g_i__w, g__w, d_budget):
        print('IID Grouping Started')
        
        (c_star, cost_star) = self.run_GroupingInternal(c, nid2_g_i__w, g__w)
        
        d_global = c_star.get_d_global(True)
        d_group = c_star.get_d_group(True)
        tau1 = 1
        if d_group < d_global:
            tau2 = max( int((d_budget + d_group - d_global) / d_group), 1 )
        else:
            tau2 = 2
        print('Final cost_star=%.4f, tau1=%d, tau2=%d' % (cost_star, tau1, tau2))
        
        print('IID Grouping Finished')
        return c_star, tau1, tau2
    
    def run_GroupingInternal(self, c, nid2_g_i__w, g__w):
        # 최적 초기화
        c_star = c.clone()
        cost_star = self.getCost(c_star, nid2_g_i__w, g__w)
        print('Initial cost_star=%.4f' % (cost_star))
        self.fwCost.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), 0, cost_star, cost_star])
        
        cntSteady = 0
        while cntSteady < GROUPING_MAX_STEADY_STEPS:
            cost_cur = self.iterate_Grouping(c, nid2_g_i__w, g__w)
            if cost_star > cost_cur:
                (c_star, cost_star) = (c.clone(), cost_cur)
                cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
            else:
                cntSteady += 1
            print('cntSteady=%d, cost_star=%.4f, cost_cur=%.4f' % (cntSteady, cost_star, cost_cur))
            self.fwCost.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), cntSteady, cost_star, cost_cur])
        return (c_star, cost_star)
    
    def getCost(self, c, nid2_g_i__w, g__w):
        return c.get_DELTA(nid2_g_i__w, g__w)
    
    def checkIntegrity(self, c, z_candidate):
        nids_byGid = fl_data.to_nids_byGid(z_candidate)
        numClassesPerGroup = np.mean([ len(np.unique(np.concatenate([c.D_byNid[nid]['y'] for nid in nids]))) for nids in nids_byGid ])
        # 그룹의 데이터 분포가 IID 인지 검사
        return numClassesPerGroup == self.model.numClasses
    
    def iterate_Grouping(self, c, nid2_g_i__w, g__w):
        cost_cur = self.getCost(c, nid2_g_i__w, g__w)

        # Iteration 마다 탐색 인덱스 셔플
        idx_Nid1 = np.arange(len(c.get_N()))
        np.random.shuffle(idx_Nid1)
        idx_Nid1 = idx_Nid1[:GROUPING_NUM_SAMPLE_NODE]

        idx_Nid2 = np.arange(len(c.get_N()))
        np.random.shuffle(idx_Nid2)
        idx_Nid2 = idx_Nid2[:GROUPING_NUM_SAMPLE_NODE]
        
        z = c.z
        for i in idx_Nid1:
            for j in idx_Nid2:
                # 목적지 그룹이 이전 그룹과 같을 경우 무시
                if z[i] == z[j]: continue
                    
                # 그룹 멤버쉽 변경 시도
                temp_k = z[i]
                z[i] = z[j]
                z[j] = temp_k
                
                if self.checkIntegrity(c, z) == False:
                    # 무결성을 만족 못할 경우, 다시 원래대로 그룹 멤버쉽 초기화
                    temp_k = z[i]
                    z[i] = z[j]
                    z[j] = temp_k
                    continue
                    
                # 한 그룹에 노드가 전부 없어지는 것은 있을 수 없는 경우
                if c.digest(z) == False: raise Exception(str(z))
                    
                # 다음 후보에 대한 Cost 계산
                cost_next = self.getCost(c, nid2_g_i__w, g__w)
                
                if cost_cur > cost_next:
                    # 유리할 경우 변경
                    cost_cur = cost_next
                else:
                    # 유리하지 않을 경우, 다시 원래대로 그룹 멤버쉽 초기화 (다음 Iteration 에서 Digest)
                    temp_k = z[i]
                    z[i] = z[j]
                    z[j] = temp_k
            print(i, cost_cur)
    
        # 마지막 멤버십 초기화가 발생했을 수도 있으므로, Cloud Digest 시도
        if c.digest(z) == False: raise Exception(str(z))
        return cost_cur