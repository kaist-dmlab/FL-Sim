import importlib


FedavgIcBasePackagePath = 'algorithm.fedavg-ic-base'
FedavgIcBaseModule = importlib.import_module(FedavgIcBasePackagePath)
FedavgIcBaseAlgorithm = getattr(FedavgIcBaseModule, 'Algorithm')

class Algorithm(FedavgIcBaseAlgorithm):
    
    def getAssociateCost(self, c):
        return c.get_sum_hpp_group(self.args.edgeCombineEnabled)
    
    def determineMedoidNids(self, c, nid2_g_i__w, g__w):
        return [ g.ps_nid for g in c.groups ]