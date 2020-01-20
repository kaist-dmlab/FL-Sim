from algorithm.abc import AbstractAlgorithm

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        tau1 = int(self.args.opaque1)
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName + '_' \
                + self.args.nodeType + self.args.edgeType + '_' + str(tau1)
    
    def getApprCommCostGlobal(self):
        return self.c.get_hpp_global(False) * 2
    
    def getCommTimeGlobal(self, linkSpeed):
        return self.c.get_d_global(False, linkSpeed) * 2
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'aggrType'])
        
        lr = self.args.lrInitial
        w = self.model.getParams()
        
    #     for t2 in range(int(self.args.maxEpoch/tau1)):
        tau1 = int(self.args.opaque1)
        t2 = 0
        while True:
            (w_byTime, _) = self.model.federated_train(w, self.c.get_D_is(), lr, tau1, self.c.get_p_is())
            w = w_byTime[-1]
            for t1 in range(tau1):
                self.t = t2*tau1 + t1 + 1
                if self.t % tau1 == 0:
                    aggrType = 'Global'
                else:
                    aggrType = ''
                (loss, _, _, acc) = self.model.evaluate(w_byTime[t1])
                print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f\taggrType=%s' % (self.t, loss, acc, aggrType))
                self.fwEpoch.writerow([self.t, loss, acc, aggrType])
                
                lr *= self.args.lrDecayRate
                if self.t >= self.args.maxEpoch: break;
            if self.t >= self.args.maxEpoch: break;
            t2 += 1