from algorithm.abc import AbstractAlgorithm
import fl_param
import fl_core

class Algorithm(AbstractAlgorithm):
    
    def getName(self):
        return 'FedAvg'
    
    def run(self):
        self.fwEpoch.writerow(['epoch', 'loss', 'accuracy', 'time', 'aggrType'])
        (_, testData_by1Nid, c, _) = self.initialize(self.args.nodeType, self.args.edgeType)
        
        lr = fl_param.LR_INITIAL
        w = None
        d_global = c.get_d_global(False) ; d_sum = 0
        
    #     for t2 in range(int(fl_param.MAX_EPOCH/tau1)):
        tau1 = self.args.opaque1
        t2 = 0
        while True:
            (w_byTime, _) = fl_core.federated_train(w, c.get_Data_is(), lr, tau1, c.get_D_is())
            w = w_byTime[-1]
            for t1 in range(tau1):
                t = t2*tau1 + t1 + 1
                if t % tau1 == 0:
                    d_sum += d_global
                    aggrType = 'Global'
                else:
                    aggrType = ''
                (loss, accuracy) = fl_core.evaluate(w_byTime[t1], testData_by1Nid[0])
                print('Epoch\t%d\tloss=%.3f\taccuracy=%.3f\ttime=%.4f\taggrType=%s' % (t, loss, accuracy, d_sum, aggrType))
                self.fwEpoch.writerow([t, loss, accuracy, d_sum, aggrType])
                
                lr *= fl_param.LR_DECAY_RATE
                if d_sum >= fl_param.MAX_TIME: break;
            if d_sum >= fl_param.MAX_TIME: break;
            t2 += 1