from algorithm.abc import AbstractAlgorithm
import fl_param
import fl_core

class Algorithm(AbstractAlgorithm):
    
    def getName(self):
        return 'HierFAVG'
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'time', 'aggrType'])
        (_, testData_by1Nid, c, _) = self.initialize(self.args.nodeType, self.args.edgeType)
        
        lr = fl_param.LR_INITIAL
        input_w_ks = [ None for _ in c.groups ]
        d_global = c.get_d_global(True) ; d_group = c.get_d_group(True) ; d_sum = 0
        
    #     for t3 in range(int(fl_param.MAX_EPOCH/(tau1*tau2))):
        tau1 = self.args.opaque1
        tau2 = self.args.opaque2
        t3 = 0
        while True:
            for t2 in range(tau2):
                w_k_byTime_byGid = []
                output_w_ks = []
                for k, g in enumerate(c.groups): # Group Aggregation
                    w_k = input_w_ks[k]
                    (w_k_byTime, _) = fl_core.federated_train(w_k, g.get_Data_k_is(), lr, tau1, g.get_D_k_is())
                    w_k_byTime_byGid.append(w_k_byTime)
                    output_w_ks.append(w_k_byTime[-1])
                input_w_ks = output_w_ks
                w_byTime = fl_core.federated_aggregate(w_k_byTime_byGid, c.get_D_ks()) # Global Aggregation
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
                    (loss, accuracy) = fl_core.evaluate(w_byTime[t1], testData_by1Nid[0])
                    print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f\ttime=%.4f\taggrType=%s' % (t, loss, accuracy, d_sum, aggrType))
                    self.fwEpoch.writerow([t, loss, accuracy, d_sum, aggrType])
                    
                    lr *= fl_param.LR_DECAY_RATE
                    if d_sum >= fl_param.MAX_TIME: break;
                if d_sum >= fl_param.MAX_TIME: break;
            if d_sum >= fl_param.MAX_TIME: break;
            t3 += 1
            
            w = w_byTime[-1]
            input_w_ks = [ w for _ in c.groups ]
            
            # 실험 출력용 Delta 계산
    #        (g_is__w, nid2_g_i__w) = fl_core.federated_collect_gradients(w, c.get_nid2_Data_i())
    #        g__w = np.average(g_is__w, axis=0, weights=c.get_D_is())
    #        Delta = c.get_Delta(nid2_g_i__w, g__w)
    #        print(Delta)