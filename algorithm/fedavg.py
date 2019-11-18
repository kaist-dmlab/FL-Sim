from algorithm.abc import AbstractAlgorithm

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        tau1 = int(self.args.opaque1)
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName + '_' \
                + self.args.nodeType + self.args.edgeType + '_' + str(tau1)
    
    def __init__(self, args):
        args.edgeType = 'a' # 무조건 처음 edgeType 을 all 로 고정
        super().__init__(args)
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'time', 'aggrType'])
        (trainData_by1Nid, testData_by1Nid, c) = self.getInitVars()
        
        lr = self.args.lrInitial
        w = self.model.getParams()
        d_global = c.get_d_global(False) ; d_sum = 0
        
    #     for t2 in range(int(self.args.maxEpoch/tau1)):
        tau1 = int(self.args.opaque1)
        t2 = 0
        while True:
            (w_byTime, _) = self.model.federated_train(w, c.get_Data_is(), lr, tau1, c.get_D_is())
            w = w_byTime[-1]
            for t1 in range(tau1):
                t = t2*tau1 + t1 + 1
                if t % tau1 == 0:
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
            t2 += 1