from algorithm.abc import AbstractAlgorithm

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        tau1 = int(self.args.opaque1)
        tau2 = int(self.args.opaque2)
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName + '_' \
                + self.args.nodeType + self.args.edgeType + '_' + str(tau1) + '_' + str(tau2)
    
    def getApprCommCostGroup(self, c):
        return c.get_max_hpp_group(self.args.edgeCombineEnabled) * 2
    
    def getApprCommCostGlobal(self, c):
        return c.get_hpp_global(self.args.edgeCombineEnabled) * 2
    
    def getCommTimeGroup(self, c, dataSize, linkSpeed):
        return c.get_d_group(self.args.edgeCombineEnabled, dataSize, linkSpeed) * 2
    
    def getCommTimeGlobal(self, c, dataSize, linkSpeed):
        return c.get_d_global(self.args.edgeCombineEnabled, dataSize, linkSpeed) * 2
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'aggrType'])
        
        lr = self.args.lrInitial
        input_w_ks = [ self.model.getParams() for _ in self.c.groups ]
        
        tau1 = int(self.args.opaque1)
        tau2 = int(self.args.opaque2)
        t3 = 0 ; time = 0
        d_local, d_group, d_global = self.getDefaultDelay()
        while True:
            for t2 in range(tau2):
                w_k_byTime_byGid = []
                output_w_ks = []
                for k, g in enumerate(self.c.groups): # Group Aggregation
                    w_k = input_w_ks[k]
                    (w_k_byTime, _) = self.model.federated_train(w_k, g.get_D_k_is(), lr, tau1, g.get_p_k_is())
                    w_k_byTime_byGid.append(w_k_byTime)
                    output_w_ks.append(w_k_byTime[-1])
                input_w_ks = output_w_ks
                w_byTime = self.model.federated_aggregate(w_k_byTime_byGid, self.c.get_p_ks()) # Global Aggregation
                for t1 in range(tau1):
                    self.epoch = t3*tau1*tau2 + t2*tau1 + t1 + 1
                    time += d_local
                    if self.epoch % tau1 == 0 and not(self.epoch % (tau1*tau2)) == 0:
                        time += d_group
                        aggrType = 'Group'
                    elif self.epoch % (tau1*tau2) == 0:
                        time += d_global
                        aggrType = 'Global'
                    else:
                        aggrType = ''
                    (loss, _, _, acc) = self.model.evaluate(w_byTime[t1])
                    print('epoch=%5d\ttime=%.3f\tloss=%.3f\taccuracy=%.3f\taggrType=%s' % (self.epoch, time, loss, acc, aggrType))
                    self.fwEpoch.writerow([self.epoch, loss, acc, aggrType])
                    
                    lr *= self.args.lrDecayRate
                    if time >= self.args.maxTime: break;
                if time >= self.args.maxTime: break;
            if time >= self.args.maxTime: break;
            t3 += 1
            
            w = w_byTime[-1]
            input_w_ks = [ w for _ in self.c.groups ]