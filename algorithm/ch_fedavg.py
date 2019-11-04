import numpy as np
import csv
from time import gmtime, strftime #strftime("%m%d_%H%M%S", gmtime()) + ' ' + 

from algorithm.abc import AbstractAlgorithm
import fl_util

CH_FEDAVG_NUM_SAMPLE_NODE = 100
CH_FEDAVG_MAX_STEADY_STEPS = 5
CH_FEDAVG_IID_GROUPING_INTERVAL = 10000

def groupRandomly(numNodes, numGroups):
    numNodesPerGroup = int(numNodes / numGroups)
    z_rand = [ k for k in range(numGroups) for _ in range(numNodesPerGroup) ]
    np.random.shuffle(z_rand)
    return z_rand

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName + '_' \
                + self.args.nodeType + '_' + self.args.edgeType + '_' + str(self.args.opaque1)
    
    def __init__(self, args):
        super().__init__(args)
        
        fileName = self.getFileName()
        self.fileDelta = open('logs/' + fileName + '_Delta.csv', 'w', newline='', buffering=1)
        self.fwDelta = csv.writer(self.fileDelta, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
    def __del__(self):
        self.fileDelta.close()
        super().__del__()
        
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'time', 'aggrType', 'numGroups', 'tau1', 'tau2'])
        (trainData_by1Nid, testData_by1Nid, _, c) = self.getInitVars()
        
        # 그룹 멤버쉽 랜덤 초기화
    #    z_rand = groupRandomly(len(c.get_N()), len(c.groups))
    #    c.digest(z_rand)
        
        lr = self.args.lrInitial
        input_w_ks = [ self.model.getParams() for _ in c.groups ]
        d_global = c.get_d_global(True) ; d_group = c.get_d_group(True) ; d_sum = 0
        d_budget = self.args.opaque1
        
    #     for t3 in range(int(self.args.maxEpoch/(tau1*tau2))):
        t = 0 ; t_prev = 0 ; tau1 = 1 ; tau2 = 1 ; t3 = 0 ;
        while True:
            for t2 in range(tau2):
                w_is = []
                w_k_byTime_byGid = []
                output_w_ks = []
                for k, g in enumerate(c.groups): # Group Aggregation
                    w_k = input_w_ks[k]
                    (w_k_byTime, w_k_is) = self.model.federated_train(w_k, g.get_Data_k_is(), lr, tau1, g.get_D_k_is())
                    w_is += w_k_is
                    w_k_byTime_byGid.append(w_k_byTime)
                    output_w_ks.append(w_k_byTime[-1])
                input_w_ks = output_w_ks
                w_byTime = self.model.federated_aggregate(w_k_byTime_byGid, c.get_D_ks()) # Global Aggregation
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
                    (loss, _, _, acc) = self.model.evaluate(w_byTime[t1], trainData_by1Nid, testData_by1Nid)
                    print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f\ttime=%.4f\taggrType=%s\tnumGroups=%d\ttau1=%d\ttau2=%d'
                          % (t, loss, acc, d_sum, aggrType, len(c.groups), tau1, tau2))
                    self.fwEpoch.writerow([t, loss, acc, d_sum, aggrType, len(c.groups), tau1, tau2])

                    lr *= self.args.lrDecayRate
    #                 if t >= self.args.maxEpoch: break
    #             if t >= self.args.maxEpoch: break
    #         if t >= self.args.maxEpoch: break
                    if d_sum >= self.args.maxTime: break;
                if d_sum >= self.args.maxTime: break;
            if d_sum >= self.args.maxTime: break;

            w = w_byTime[-1] # 출력을 위해 매 시간마다 수행했던 Global Aggregation 의 마지막 시간 값만 추출
            input_w_ks = [ w for _ in c.groups ]

            if t3 % CH_FEDAVG_IID_GROUPING_INTERVAL == 0:
                # Gradient Estimation
                (g_is__w, nid2_g_i__w) = self.model.federated_collect_gradients(w, c.get_nid2_Data_i())
    #             (g_is__w2, nid2_g_i__w2) = self.model.federated_collect_gradients2(w, c.get_nid2_Data_i())
    #             for nid in nid2_g_i__w:
    #                 print(self.model.np_modelEquals(g_is__w[nid], g_is__w2[nid]), self.model.np_modelEquals(nid2_g_i__w[nid], nid2_g_i__w2[nid]))
                g__w = np.average(g_is__w, axis=0, weights=c.get_D_is())

                # IID Grouping
                (c, tau1, tau2, d_group, d_global) = self.run_IID_Grouping(c, nid2_g_i__w, g__w, d_budget)
            t_prev = t # Mark Time
            t3 += 1
            
    def run_IID_Grouping(self, c, nid2_g_i__w, g__w, d_budget):
        print('IID Grouping Started')
        
        # 최적 초기화
        (c_star, Delta_star) = self.iterate_IID_Grouping(c, nid2_g_i__w, g__w)
        print('Initial Delta_star=%.4f' % (Delta_star))
        self.fwDelta.writerow(['time', 'cntSteady', 'Delta_star', 'Delta_cur'])
        self.fwDelta.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), 0, Delta_star, Delta_star])
        
        cntSteady = 0
        while cntSteady < CH_FEDAVG_MAX_STEADY_STEPS:
            (c, Delta) = self.iterate_IID_Grouping(c, nid2_g_i__w, g__w)
            if Delta_star > Delta:
                (c_star, Delta_star) = (c, Delta)
                cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
            else:
                cntSteady += 1
            print('cntSteady=%d, Delta_star=%.4f, Delta_cur=%.4f' % (cntSteady, Delta_star, Delta))
            self.fwDelta.writerow([strftime("%m-%d_%H:%M:%S", gmtime()), cntSteady, Delta_star, Delta])
            
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
    
    def iterate_IID_Grouping(self, c, nid2_g_i__w, g__w):
        Delta_cur = c.get_Delta(nid2_g_i__w, g__w)

        # Iteration 마다 탐색 인덱스 셔플
        idx_Nid1 = np.arange(len(c.get_N()))
        np.random.shuffle(idx_Nid1)
        idx_Nid1 = idx_Nid1[:CH_FEDAVG_NUM_SAMPLE_NODE]

        idx_Nid2 = np.arange(len(c.get_N()))
        np.random.shuffle(idx_Nid2)
        idx_Nid2 = idx_Nid2[:CH_FEDAVG_NUM_SAMPLE_NODE]

        z = c.z
        for i in idx_Nid1:
            for j in idx_Nid2:
                # 목적지 그룹이 이전 그룹과 같을 경우 무시
                if z[i] == z[j]: continue
                    
                # 그룹 멤버쉽 변경 시도
                temp_k = z[i]
                z[i] = z[j]
                z[j] = temp_k
                
                nids_byGid = fl_util.to_nids_byGid(z)
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