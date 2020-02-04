import importlib
import numpy as np

from cloud.cloud import Cloud
import fl_data

FedavgIcBasePackagePath = 'algorithm.fedavg-ic-abc'
FedavgIcBaseModule = importlib.import_module(FedavgIcBasePackagePath)
FedavgIcBaseAlgorithm = getattr(FedavgIcBaseModule, 'Algorithm')

GROUPING_MAX_STEADY_STEPS = 1

NUM_NORM_CONST_SAMPLES = 100
NC_TYPE = 'mu' # mu, min-max, z-score
# min-max are z-score are not appropriate for the normalization of multi-objective optimization
# because minus value may occur and interfere with cost search process

class Algorithm(FedavgIcBaseAlgorithm):
        
    def runGrouping(self, c_edge, nid2_g_i__w, g__w):
        print('Comm-IID Grouping Started')
        
        # 최적 초기화
        c_star = None
        cost_star = float('inf')
        
        # 그룹 개수 바꿔가면서 탐색
        cntSteady = 0
        numGroups = self.args.numGroups # numGroups = 1 일 때는, DELTA 가 0 이 되므로 다른 numGroups 경우와 Cost 비교 불가
        while cntSteady < GROUPING_MAX_STEADY_STEPS and numGroups <= self.args.numGroups:
            # Cost Normalization Constant 샘플링
            iid_costs = [] ; comm_final_costs = []
            iid_k_costs = [] ; comm_k_costs = []
            for _ in range(NUM_NORM_CONST_SAMPLES):
                c_sample = Cloud(c_edge.topology, c_edge.D_byNid, numGroups)
                
                # Get Global Associate Cost
                medoidNids = np.random.choice(np.arange(self.args.numNodes), size=numGroups, replace=False)
                medoidNids_byGid = [ [nid] for nid in medoidNids ]
                if c_sample.digest(medoidNids_byGid, nid2_g_i__w, g__w) == False: raise Exception(str(medoidNids_byGid))
                    
                randNids = np.arange(self.args.numNodes)
                np.random.shuffle(randNids)
                
                z = fl_data.to_z(self.args.numNodes, c_sample.nids_byGid)
                for nid in randNids:
                    if nid in medoidNids: continue
                        
                    k = np.random.randint(numGroups)
                    z[nid] = k
                    nids_byGid = fl_data.to_nids_byGid(z)
                    if c_sample.digest(nids_byGid, nid2_g_i__w, g__w) == False: raise Exception(str(nids_byGid))
                    iid_costs.append(c_sample.get_DELTA())
                    comm_final_costs.append(c_sample.get_sum_hpp_group(False))
                    
                # Get Group Medoid Cost
                for g in c_sample.groups:
                    N_k = g.get_N_k()
                    costs = []
                    for nid1 in N_k:
                        p_i = c_sample.get_p_is()[nid1]
                        iid_i = np.linalg.norm(nid2_g_i__w[nid1] - g__w)
                        comm_i = sum( c_sample.topology.getDistance(nid1, nid2) for nid2 in N_k )
                        iid_k_costs.append(p_i*iid_i)
                        comm_k_costs.append(comm_i)
            self.nc_iid_mu = np.mean(iid_costs)
            self.nc_iid_std = np.std(iid_costs)
            self.nc_iid_min = min(iid_costs)
            self.nc_iid_max = max(iid_costs)
            
            self.nc_comm_mu = np.mean(comm_final_costs)
            self.nc_comm_std = np.std(comm_final_costs)
            self.nc_comm_min = min(comm_final_costs)
            self.nc_comm_max = max(comm_final_costs)
            
            self.nc_iid_k_mu = np.mean(iid_k_costs)
            self.nc_iid_k_std = np.std(iid_k_costs)
            self.nc_iid_k_min = min(iid_k_costs)
            self.nc_iid_k_max = max(iid_k_costs)
            
            self.nc_comm_k_mu = np.mean(comm_k_costs)
            self.nc_comm_k_std = np.std(comm_k_costs)
            self.nc_comm_k_min = min(comm_k_costs)
            self.nc_comm_k_max = max(comm_k_costs)
            print('Cost Norm Constant:', self.nc_iid_min, self.nc_iid_max, self.nc_comm_min, self.nc_comm_max, \
                  self.nc_iid_k_min, self.nc_iid_k_max, self.nc_comm_k_min, self.nc_comm_k_max)
            
            print('numGroups:', numGroups)
            c_cur = Cloud(c_edge.topology, c_edge.D_byNid, numGroups)
            z_rand = fl_data.groupRandomly(self.args.numNodes, numGroups)
            nids_byGid = fl_data.to_nids_byGid(z_rand)
            c_cur.digest(nids_byGid, nid2_g_i__w, g__w)
            c_cur, cost_cur = self.runKMedoidsGrouping(c_cur, nid2_g_i__w, g__w)
            if self.isGreaterThan(cost_star, cost_cur):
                c_star, cost_star = c_cur, cost_cur
                cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
            else:
                cntSteady += 1
            numGroups += 1
        
        default_linkSpeed = self.args.linkSpeeds[0]
        d_group = self.getCommTimeGroup(c_star, self.model.size, default_linkSpeed)
        d_global = self.getCommTimeGlobal(c_star, self.model.size, default_linkSpeed)
        self.setDefaultCommDelay(d_group, d_global)
        
        print('Final cost_star=%.3f, numGroups=%d, d_group=%.3f, d_global=%.3f' % (cost_star, len(c_star.groups), d_group, d_global))
        print('Comm-IID Grouping Finished')
        return c_star
    
    def getAssociateCost(self, c, nid, medoidNid):
        if self.nc_comm_mu == 0 or self.nc_iid_mu == 0:
            raise Exception(str(self.nc_iid_mu), str(self.nc_comm_mu))
        cost_iid = c.get_DELTA() ; cost_comm = c.get_sum_hpp_group(False)
        if NC_TYPE == 'mu':
            return cost_iid/self.nc_iid_mu + cost_comm/self.nc_comm_mu
        elif NC_TYPE == 'min-max':
            return (cost_iid-self.nc_iid_min)/(self.nc_iid_max-self.nc_iid_min) \
                    + (cost_comm-self.nc_comm_min)/(self.nc_comm_max-self.nc_comm_min)
        elif NC_TYPE == 'z-score':
            return (cost_iid-self.nc_iid_mu)/self.nc_iid_std + (cost_comm-self.nc_comm_mu)/self.nc_comm_std
        else:
            raise Exception(NC_TYPE)
            
    def determineMedoidNids(self, c, nid2_g_i__w, g__w):
        medoidNids = []
        for g in c.groups:
            N_k = g.get_N_k()
            costs = []
            for nid1 in N_k:
                p_i = c.get_p_is()[nid1]
                iid_i = np.linalg.norm(nid2_g_i__w[nid1] - g__w)
                comm_i = sum( c.topology.getDistance(nid1, nid2) for nid2 in N_k )
                if NC_TYPE == 'mu':
                    costs.append( p_i*iid_i/self.nc_iid_k_mu + comm_i/self.nc_comm_k_mu )
                elif NC_TYPE == 'min-max':
                    costs.append( (p_i*iid_i-self.nc_iid_k_min)/(self.nc_iid_k_max-self.nc_iid_k_min) \
                                 + (comm_i-self.nc_comm_k_min)/(self.nc_comm_k_max-self.nc_comm_k_min) )
                elif NC_TYPE == 'z-score':
                    costs.append( (p_i*iid_i-self.nc_iid_k_mu)/self.nc_iid_k_std + (comm_i-self.nc_comm_k_mu)/self.nc_comm_k_std )
                else:
                    raise Exception(NC_TYPE)
            medoidNids.append(N_k[ np.argmin(costs) ])
        return medoidNids