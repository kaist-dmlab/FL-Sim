from algorithm.abc import AbstractAlgorithm

class Algorithm(AbstractAlgorithm):
    
    def getName(self):
        return 'CGD'
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy'])
        (trainData_by1Nid, testData_by1Nid, _, _) = self.getInitVars()
        
        lr = self.args.lrInitial
        w = self.model.getParams()
        
        for t in range(1, self.args.maxEpoch):
            (w_byTime, _) = self.model.federated_train(w, trainData_by1Nid, lr, 1, [1])
            w = w_byTime[0]
            (loss, accuracy) = self.model.evaluate(w, testData_by1Nid[0])
            print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f' % (t, loss, accuracy))
            self.fwEpoch.writerow([t, loss, accuracy])
            lr *= self.args.lrDecayRate