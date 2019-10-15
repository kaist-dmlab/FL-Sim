from algorithm.abc import AbstractAlgorithm
import fl_param
import fl_core

class Algorithm(AbstractAlgorithm):
    
    def getName(self):
        return 'CGD'
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy'])
        (trainData_by1Nid, testData_by1Nid, _, _) = self.initialize(self.args.nodeType, self.args.edgeType)
        
        lr = fl_param.LR_INITIAL
        w = None
        
        for t in range(1, fl_param.MAX_EPOCH):
            (w_byTime, _) = fl_core.federated_train(w, trainData_by1Nid, lr, 1, [1])
            w = w_byTime[0]
            (loss, accuracy) = fl_core.evaluate(w, testData_by1Nid[0])
            print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f' % (t, loss, accuracy))
            self.fwEpoch.writerow([t, loss, accuracy])
            lr *= fl_param.LR_DECAY_RATE