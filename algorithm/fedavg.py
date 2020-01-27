from algorithm.abc import AbstractAlgorithm

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        tau1 = int(self.args.opaque1)
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName + '_' \
                + self.args.nodeType + self.args.edgeType + '_' + str(tau1)
    
    def getApprCommCostGlobal(self):
        return self.c.get_hpp_global(False) * 2
    
    def getCommTimeGlobal(self, c, dataSize, linkSpeed):
        return c.get_d_global(False, dataSize, linkSpeed) * 2
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'aggrType'])
        
        lr = self.args.lrInitial
        w = self.model.getParams()

        tau1 = int(self.args.opaque1)
        t2 = 0 ; time = 0
        while True:
            (w_byTime, _) = self.model.federated_train(w, self.c.get_D_is(), lr, tau1, self.c.get_p_is())
            w = w_byTime[-1]
            for t1 in range(tau1):
                self.epoch = t2*tau1 + t1 + 1
                time += self.d_local
                if self.epoch % tau1 == 0:
                    time += self.d_global
                    aggrType = 'Global'
                else:
                    aggrType = ''
                (loss, _, _, acc) = self.model.evaluate(w_byTime[t1])
                print('epoch=%5d\ttime=%.3f\tloss=%.3f\taccuracy=%.3f\taggrType=%s' % (self.epoch, time, loss, acc, aggrType))
                self.fwEpoch.writerow([self.epoch, loss, acc, aggrType])
                
                lr *= self.args.lrDecayRate
                if time >= self.args.maxTime: break;
            if time >= self.args.maxTime: break;
            t2 += 1