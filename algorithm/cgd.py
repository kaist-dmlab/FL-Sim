from algorithm.abc import AbstractAlgorithm

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy'])
        (trainData_by1Nid, testData_by1Nid, _, _) = self.getInitVars()
        
        lr = self.args.lrInitial
        w = self.model.getParams()
        
        for t in range(1, self.args.maxEpoch):
            (w_byTime, _) = self.model.federated_train(w, trainData_by1Nid, lr, 1, [1])
            w = w_byTime[0]
            (loss, _, _, acc) = self.model.evaluate(w, trainData_by1Nid, testData_by1Nid)
            print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f' % (t, loss, acc))
            self.fwEpoch.writerow([t, loss, acc])
            lr *= self.args.lrDecayRate