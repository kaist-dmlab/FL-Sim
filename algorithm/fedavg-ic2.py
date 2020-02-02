import importlib
import numpy as np

import fl_data

FedavgIcBasePackagePath = 'algorithm.fedavg-ic-abc'
FedavgIcBaseModule = importlib.import_module(FedavgIcBasePackagePath)
FedavgIcBaseAlgorithm = getattr(FedavgIcBaseModule, 'Algorithm')

# Epsilon-Constraint Method
class Algorithm(FedavgIcBaseAlgorithm):
    
    def associate(self, c, nid2_g_i__w, g__w, medoidNids):
        c = c.clone(nid2_g_i__w, g__w)
        medoidNids_byGid = [ [nid] for nid in medoidNids ]
        if c.digest(medoidNids_byGid, nid2_g_i__w, g__w) == False: raise Exception(str(medoidNids_byGid))
        
        # Shuffle index every iteration
        randNids = np.arange(self.args.numNodes)
        np.random.shuffle(randNids)
        
        z = fl_data.to_z(self.args.numNodes, c.nids_byGid)
        for i in randNids:
            # Medoid 일 경우 무시
            if i in medoidNids: continue
                
            # Search for candidates with the same minimum cost
            iid_costs = [] ; comm_costs = []
            for k, g in enumerate(c.groups):
                # 목적지 그룹이 이전 그룹과 같을 경우 무시
                if z[i] == k: continue
                    
                # 그룹 멤버쉽 변경 시도
                z[i] = k
                
                # Cloud Digest 시도
                nids_byGid = fl_data.to_nids_byGid(z)
                if c.digest(nids_byGid, nid2_g_i__w, g__w) == False: raise Exception(str(z), str(nids_byGid))
                    
                # 다음 후보에 대한 Cost 계산
                iid_costs.append(self.getIidCost(c))
                comm_costs.append(self.getCommCost(c))
            iid_costs = np.array(iid_costs)
            
            # IID Cost 를 Epsilon-Constraint 로 설정
            # Comm Cost 는 fattree 의 경우 Cost 변화가 다양하지 않아서, percentile 로 나눠도 의미가 없으므로 Constraint 불가
            # e.g.) [4, 4, 6, 6, 6, 6, 6] => np.percentile(, 75) = 6 => 결과적으로 Filtering 이 안됨
            low_iid_ks = np.where(iid_costs <= np.percentile(iid_costs, 75))[0]
            if len(low_iid_ks) == 0:
                low_iid_ks = np.arange(len(iid_costs))
            
            k_and_comm_cost_list = [ (low_iid_k, comm_costs[low_iid_k]) for low_iid_k in low_iid_ks ]
            k_and_comm_cost_list.sort(key=lambda x:x[1]) # sort in ascending order of iid cost
            min_k = k_and_comm_cost_list[0][0]
            z[i] = min_k # 다음 Iteration 에서 Digest
            iid_cost_min = iid_costs[min_k]
            comm_cost_min = comm_costs[min_k]
            print('nid: %3d\tcurrent cost: %.3f' % (i, comm_cost_min))
            
        # 마지막 멤버십 초기화가 발생했을 수도 있으므로, Cloud Digest 시도
        nids_byGid = fl_data.to_nids_byGid(z)
        if c.digest(nids_byGid, nid2_g_i__w, g__w) == False: raise Exception(str(z), str(nids_byGid))
            
        for k, g in enumerate(c.groups):
            p_k = c.get_p_ks()[k]
            print(g.ps_nid, g.get_N_k(), p_k*g.get_DELTA_k())
        print('DELTA: ', c.get_DELTA())
        return c, comm_cost_min
    
    def getIidCost(self, c):
        return c.get_DELTA()
    
    def getCommCost(self, c):
        return c.get_sum_hpp_group(self.args.edgeCombineEnabled)
    
    def getAssociateCost(self, c):
        pass
    
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