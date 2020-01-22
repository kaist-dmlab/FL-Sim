from algorithm.abc import AbstractAlgorithm

class Algorithm(AbstractAlgorithm):
    
    def getFileName(self):
        return self.args.modelName + '_' + self.args.dataName + '_' + self.args.algName
    
    def getFlopOpsPerEpoch(self):
        return len(self.trainData_by1Nid[0]['x']) * self.model.flopOps
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'aggrType'])
        
        lr = self.args.lrInitial
        w = self.model.getParams()
        
        time = 0
        for self.epoch in range(1, self.args.maxEpoch):
            time += self.d_local
            (w_byTime, _) = self.model.federated_train(w, self.trainData_by1Nid, lr, 1, [1])
            w = w_byTime[0]
            (loss, _, _, acc) = self.model.evaluate(w)
            print('epoch=%5d\ttime=%.3f\tloss=%.3f\taccuracy=%.3f\taggrType=%s' % (self.epoch, time, loss, acc, ''))
            self.fwEpoch.writerow([self.epoch, loss, acc, ''])
            lr *= self.args.lrDecayRate