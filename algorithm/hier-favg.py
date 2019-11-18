from algorithm.abc import AbstractAlgorithm

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        tau1 = int(self.args.opaque1)
        tau2 = int(self.args.opaque2)
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName + '_' \
                + self.args.nodeType + self.args.edgeType + '_' + str(tau1) + '_' + str(tau2)
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'time', 'aggrType'])
        (trainData_by1Nid, testData_by1Nid, c) = self.getInitVars()
        
        lr = self.args.lrInitial
        input_w_ks = [ self.model.getParams() for _ in c.groups ]
        d_global = c.get_d_global(True) ; d_group = c.get_d_group(True) ; d_sum = 0
        
    #     for t3 in range(int(self.args.maxEpoch/(tau1*tau2))):
        tau1 = int(self.args.opaque1)
        tau2 = int(self.args.opaque2)
        t3 = 0
        while True:
            for t2 in range(tau2):
                w_k_byTime_byGid = []
                output_w_ks = []
                for k, g in enumerate(c.groups): # Group Aggregation
                    w_k = input_w_ks[k]
                    (w_k_byTime, _) = self.model.federated_train(w_k, g.get_Data_k_is(), lr, tau1, g.get_D_k_is())
                    w_k_byTime_byGid.append(w_k_byTime)
                    output_w_ks.append(w_k_byTime[-1])
                input_w_ks = output_w_ks
                w_byTime = self.model.federated_aggregate(w_k_byTime_byGid, c.get_D_ks()) # Global Aggregation
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
                    (loss, _, _, acc) = self.model.evaluate(w_byTime[t1], trainData_by1Nid, testData_by1Nid)
                    print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f\ttime=%.4f\taggrType=%s' % (t, loss, acc, d_sum, aggrType))
                    self.fwEpoch.writerow([t, loss, acc, d_sum, aggrType])
                    
                    lr *= self.args.lrDecayRate
                    if d_sum >= self.args.maxTime: break;
                if d_sum >= self.args.maxTime: break;
            if d_sum >= self.args.maxTime: break;
            t3 += 1
            
            w = w_byTime[-1]
            input_w_ks = [ w for _ in c.groups ]
            
            # 실험 출력용 Delta 계산
    #        (g_is__w, nid2_g_i__w) = self.model.federated_collect_gradients(w, c.get_nid2_Data_i())
    #        g__w = np.average(g_is__w, axis=0, weights=c.get_D_is())
    #        Delta = c.get_Delta(nid2_g_i__w, g__w)
    #        print(Delta)