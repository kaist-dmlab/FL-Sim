from algorithm.abc import AbstractAlgorithm

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy'])
        
        lr = self.args.lrInitial
        w = self.model.getParams()
        
        for self.t in range(1, self.args.maxEpoch):
            (w_byTime, _) = self.model.federated_train(w, self.trainData_by1Nid, lr, 1, [1])
            w = w_byTime[0]
            (loss, _, _, acc) = self.model.evaluate(w)
            print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f' % (self.t, loss, acc))
            self.fwEpoch.writerow([self.t, loss, acc])
            lr *= self.args.lrDecayRate