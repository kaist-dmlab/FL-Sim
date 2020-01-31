import importlib
import numpy as np

FedavgIcBasePackagePath = 'algorithm.fedavg-ic-base'
FedavgIcBaseModule = importlib.import_module(FedavgIcBasePackagePath)
FedavgIcBaseAlgorithm = getattr(FedavgIcBaseModule, 'Algorithm')

class Algorithm(FedavgIcBaseAlgorithm):
    
    def getAssociateCost(self, c):
        return c.get_DELTA()
    
    def determineMedoidNids(self, c, nid2_g_i__w, g__w):
        medoidNids = []
        for g in c.groups:
            N_k = g.get_N_k()
            costs = []
            for nid in N_k:
                p_i = c.get_p_is()[nid]
                iid_i = np.linalg.norm(nid2_g_i__w[nid] - g__w)
                costs.append(p_i * iid_i)
            medoidNids.append(N_k[ np.argmin(costs) ])
        return medoidNids