import numpy as np

import fl_const
import fl_data

MAX_STEADY_STEPS = 30 # FIXME: too much

class Cloud:
    
    def __init__(self, topology, D_byNid, numGroups):
        self.topology = topology
        self.D_byNid = D_byNid
        numTotalClasses = len(np.unique(np.concatenate([ D_byNid[nid]['y'] for nid in range(len(D_byNid)) ])))
        cid2_pc = np.zeros(numTotalClasses, dtype=np.int32)
        nid2_cid2_pc_is = {}
        nid2_cid2_numClasses_is = {}
        for nid in range(len(D_byNid)):
            cid2_numClasses_i = np.zeros(numTotalClasses, dtype=np.int32)
            for j in range(len(D_byNid[nid]['x'])):
                cid = self.D_byNid[nid]['y'][j]
                cid2_pc[cid] += 1
                cid2_numClasses_i[cid] += 1
            nid2_cid2_numClasses_is[nid] = cid2_numClasses_i
            nid2_cid2_pc_is[nid] = cid2_numClasses_i / sum(cid2_numClasses_i)
        cid2_pc = cid2_pc / sum(cid2_pc)
        
        self.groups = [ Group(k, topology, D_byNid, cid2_pc, nid2_cid2_pc_is, nid2_cid2_numClasses_is) for k in range(numGroups) ]
        self.ready = False
    
    def __eq__(self, other):
        if self.groups != other.groups:
            raise Exception(str(self.groups), str(other.groups))
        if self.ready != other.ready:
            raise Exception(str(self.ready), str(other.ready))
        return True
    
    def clone(self, nid2_g_i__w=None, g__w=None):
        c_cloned = Cloud(self.topology, self.D_byNid, len(self.groups))
        c_cloned.set_p_is(self.get_p_is())
        c_cloned.digest(self.nids_byGid, nid2_g_i__w, g__w)
        return c_cloned
    
    def invalidate(self):
        self.ready = False
        for g in self.groups:
            g.invalidate()
            
    def get_N(self): # 전체 노드 집합
        if self.ready == False: raise Exception
        return self.N
    
    def get_p_ks(self): # 그룹 별 데이터 크기 집합
        if self.ready == False: raise Exception
        return self.p_ks
        
    def set_p_is(self, p_is):
        for g in self.groups:
            g.set_p_is(p_is)
        self.ready = False
    
    def get_p_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_p_i.values()) # nid 순으로 정렬된 값 반환
    
    def get_nid2_D_i(self): # 전체 노드 별 데이터 집합
        if self.ready == False: raise Exception
        return self.nid2_D_i
    
    def get_D_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_D_i.values()) # nid 순으로 정렬된 값 반환
    
    def get_d_global(self, edgeCombineEnabled, dataSize=fl_const.DEFAULT_PACKET_SIZE, linkSpeed=fl_const.DEFAULT_LINK_SPEED):
        if self.ready == False: raise Exception
        return self.topology.getDelay([ [nid, self.ps_nid] for nid in self.N ], edgeCombineEnabled, dataSize, linkSpeed)
    
    def get_d_group(self, edgeCombineEnabled, dataSize=fl_const.DEFAULT_PACKET_SIZE, linkSpeed=fl_const.DEFAULT_LINK_SPEED):
        if self.ready == False: raise Exception
        commPairs = []
        for g in self.groups:
            commPairs += [ [nid, g.ps_nid] for nid in g.get_N_k() ]
        return self.topology.getDelay(commPairs, edgeCombineEnabled, dataSize, linkSpeed)
    
    def get_hpp_global(self, edgeCombineEnabled):
        if self.ready == False: raise Exception
        return self.topology.getSumOfHopsPerPacket([ [nid, self.ps_nid] for nid in self.N ], edgeCombineEnabled)
    
    def get_max_hpp_group(self, edgeCombineEnabled):
        if self.ready == False: raise Exception
        return max( self.topology.getSumOfHopsPerPacket([ [nid, g.ps_nid] for nid in g.get_N_k() ], edgeCombineEnabled) for g in self.groups )
    
    def get_sum_hpp_group(self, edgeCombineEnabled):
        if self.ready == False: raise Exception
        return sum( self.topology.getSumOfHopsPerPacket([ [nid, g.ps_nid] for nid in g.get_N_k() ], edgeCombineEnabled) for g in self.groups )
    
    def get_delay_group(self, nid2delay):
        if self.ready == False: raise Exception
        return max( sum( nid2delay[nid] for nid in g.get_N_k() ) for g in self.groups )
    
    def get_delta(self):
        if self.ready == False: raise Exception
        delta_ks = [ g.get_delta_k() for g in self.groups ]
        delta = np.average(delta_ks, weights=self.get_p_ks())
        return delta
    
    def get_Delta(self, nid2_g_i__w, g__w):
        if self.ready == False: raise Exception
        Delta_is = [ np.linalg.norm(nid2_g_i__w[nid] - g__w) for nid in self.get_N() ]
        Delta = np.average(Delta_is, weights=self.get_p_is())
        return Delta
    
    def get_DELTA(self):
        if self.ready == False: raise Exception
        DELTA_ks = [ g.get_DELTA_k() for g in self.groups ]
        DELTA = np.average(DELTA_ks, weights=self.get_p_ks())
        return DELTA
    
    def get_emd(self):
        if self.ready == False: raise Exception
        emd_ks = [ g.get_emd_k() for g in self.groups ]
        emd = np.average(emd_ks, weights=self.get_p_ks())
        return emd
    
    def get_EMD(self):
        if self.ready == False: raise Exception
        EMD_ks = [ g.get_EMD_k() for g in self.groups ]
        EMD = np.average(EMD_ks, weights=self.get_p_ks())
        return EMD
    
    def digest(self, nids_byGid, nid2_g_i__w=None, g__w=None, debugging=False):
        if nids_byGid == None:
            return False
        else:
            self.nids_byGid = nids_byGid
            for k, N_k in enumerate(nids_byGid):
                self.groups[k].set_N_k(N_k)
                
            # digest in groups
            self.N = []
            self.D = 0
            self.p_ks = []
            self.nid2_p_i = {}
            self.nid2_D_i = {}
            for g in self.groups:
                g.digest(nid2_g_i__w, g__w, debugging)
                self.N += g.get_N_k()
                self.D += g.get_p_k()
                self.p_ks.append(g.get_p_k())
                self.nid2_p_i.update(g.get_nid2_p_k_i()) # p_is 로 변환
                self.nid2_D_i.update(g.get_nid2_D_k_i())
            self.N = sorted(self.N)
            
            # determine global parameter server
            self.ps_nid = self.N[ np.argmin([ sum( self.topology.getDistance(nid1, nid2) for nid2 in self.N ) for nid1 in self.N ]) ]
            
            # determine group parameter servers
            for g in self.groups:
                g.ps_nid = g.N_k[ np.argmin([ sum( self.topology.getDistance(nid1, nid2) for nid2 in g.N_k ) for nid1 in g.N_k ]) ]
#             numGroups = len(self.groups)
#             maxGroupsPerEdge = np.ceil(numGroups / self.topology.numEdges)
#             cost_star = float('inf')
#             cntSteady = 0
#             while cntSteady < MAX_STEADY_STEPS:
#                 eid2Cnt = { eid:0 for eid in range(self.topology.numEdges) }
#                 cur_ps_nids = []
#                 cur_ps_eids = []
#                 cur_cost_ks = []
                
#                 # iterate groups in a random order and sample a node for the parameter server
#                 # such that it minimizes the hop distance cost and maximizes the inter-cluster similarity
#                 randGids = np.arange(numGroups)
#                 np.random.shuffle(randGids)
#                 for k in randGids:
#                     g = self.groups[k]
#                     nidAndCostList = []
#                     for nid1 in g.N_k:
#                         cost = sum( self.topology.getDistance(nid1, nid2) for nid2 in g.N_k )
#                         nidAndCostList.append((nid1, cost))
#                     nidAndCostList.sort(key=lambda x:x[1]) # sort in ascending order of cost
                    
#                     # iterate nodes in the group to pick a new parameter server
#                     cur_ps_nid = None
#                     for cur_nid, cur_cost in nidAndCostList:
#                         cur_eid = self.topology.getEid(cur_nid)
                        
#                         # check if the current candidate maximizes the inter-cluster similarity
#                         if eid2Cnt[cur_eid]+1 > maxGroupsPerEdge:
#                             # if it doesn't, move on to the next candidate
#                             continue
#                         else:
#                             # if it does, pick as a new parameter server
#                             eid2Cnt[cur_eid] += 1
#                             cur_ps_nid = cur_nid
#                             cur_ps_nids.append(cur_nid)
#                             cur_ps_eids.append(cur_eid)
#                             cur_cost_ks.append(cur_cost)
#                             break
#                     # break if no node satisfied the condition
#                     if cur_ps_nid == None: break
                        
#                 # check for improvement
#                 cost_cur = float('inf') if len(cur_ps_nids) < numGroups else max(cur_cost_ks)
#                 if cost_star > cost_cur:
#                     cost_star = cost_cur
#                     cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
                    
#                     # assign parameter servers
#                     for idxNormal, idxRand in enumerate(randGids):
#                         g = self.groups[idxRand]
#                         g.ps_nid = cur_ps_nids[idxNormal]
# #                     print('Determining parameter server, cntSteady=%2d, cost_star=%.3f, parameter servers=%s' % \
# #                           (cntSteady, cost_star, str(sorted(cur_ps_nids))))
# #                     print()
#                 else:
#                     cntSteady += 1
            
#             # Check if no candidate found
#             if cost_star == float('inf'): raise Exception(str(MAX_STEADY_STEPS))
                
            # Set flag
            self.ready = True
            return True

class Group:
    
    def __init__(self, k, topology, D_byNid, cid2_pc, nid2_cid2_pc_is, nid2_cid2_numClasses_is):
        self.k = k
        self.topology = topology
        self.D_byNid = D_byNid
        self.p_is = None
        self.cid2_pc = cid2_pc
        self.nid2_cid2_pc_is = nid2_cid2_pc_is
        self.nid2_cid2_numClasses_is = nid2_cid2_numClasses_is
        self.N_k = []
        self.ps_nid = None
        self.ready = False
        
    def __eq__(self, other):
        if self.k != other.k:
            raise Exception(str(self.k), str(other.k))
        if self.p_is != other.p_is:
            raise Exception(str(self.p_is), str(other.p_is))
        if self.N_k != other.N_k:
            raise Exception(str(self.N_k), str(other.N_k))
        if self.ready != other.ready:
            raise Exception(str(self.ready), str(other.ready))
        return True
    
    def invalidate(self):
        self.ready = False
        
    def set_p_is(self, p_is):
        self.p_is = p_is
        self.ready = False
        
    def set_N_k(self, N_k):
        if not(self.N_k == N_k):
            self.N_k = N_k
            self.ready = False
            
    def get_N_k(self): # 그룹 노드 집합
        if self.ready == False: raise Exception
        return self.N_k
    
    def get_p_k(self): # 그룹 데이터 크기
        if self.ready == False: raise Exception
        return self.p_k
    
    def get_nid2_p_k_i(self): # 그룹 노드 별 데이터 크기 집합
        if self.ready == False: raise Exception
        return self.nid2_p_k_i
    
    def get_p_k_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_p_k_i.values()) # nid 순으로 정렬된 값 반환
    
    def get_nid2_D_k_i(self): # 그룹 노드 별 데이터 집합
        if self.ready == False: raise Exception
        return self.nid2_D_k_i
    
    def get_D_k_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_D_k_i.values()) # nid 순으로 정렬된 값 반환
    
    def get_delta_k(self):
        if self.ready == False: raise Exception
        return self.delta_k
    
    def get_DELTA_k(self):
        if self.ready == False: raise Exception
        return self.DELTA_k
        
    def calcEMD(self, a, b):
        if len(a) != len(b): raise Exception(len(a), len(b))
        return sum([ abs(a[i] - b[i]) for i in range(len(a)) ])
    
    def get_emd_k(self):
        if self.ready == False: raise Exception
        return self.emd_k
    
    def get_EMD_k(self):
        if self.ready == False: raise Exception
        return self.EMD_k
    
    def digest(self, nid2_g_i__w=None, g__w=None, debugging=False):
        if self.ready == True: return # Group 에 변화가 없을 때 연산되는 것을 방지
        self.p_k = 0
        p_k_is = []
        self.nid2_p_k_i = {}
        self.nid2_D_k_i = {}
        for nid in self.N_k:
            D_k_i = self.D_byNid[nid]
            if self.p_is == None: # Weight p 미설정 시, 데이터 개수로 가중치
                p_k_i = len(D_k_i['x'])
            else:
                p_k_i = self.p_is[nid]
            self.p_k += p_k_i
            p_k_is.append(p_k_i)
            self.nid2_p_k_i[nid] = p_k_i
            self.nid2_D_k_i[nid] = D_k_i
        
        if nid2_g_i__w != None:
#             g_k_is__w = [ nid2_g_i__w[nid] for nid in self.N_k ]
#             g_k__w = np.average(g_k_is__w, axis=0, weights=p_k_is)
#             delta_k_is = [ np.linalg.norm(nid2_g_i__w[nid] - g_k__w) for nid in self.N_k ]
#             self.delta_k = np.average(delta_k_is, weights=p_k_is)
            
            g_k_is__w = [ nid2_g_i__w[nid] for nid in self.N_k ]
            g_k__w = np.average(g_k_is__w, axis=0, weights=p_k_is)
            self.DELTA_k = np.linalg.norm(g_k__w - g__w)
            
#         numTotalClasses = len(self.cid2_pc)
#         cid2_pc_k = np.zeros(numTotalClasses, dtype=np.int32)
#         for nid in self.N_k:
#             cid2_pc_k += self.nid2_cid2_numClasses_is[nid]
#         cid2_pc_k = cid2_pc_k / sum(cid2_pc_k)
#         emd_k_is = [ self.calcEMD(self.nid2_cid2_pc_is[nid], cid2_pc_k) for nid in self.N_k ]
#         self.emd_k = np.average(emd_k_is, weights=p_k_is)
        
#         cid2_pc_k = np.zeros(numTotalClasses, dtype=np.int32)
#         for nid in self.N_k:
#             cid2_pc_k += self.nid2_cid2_numClasses_is[nid]
#         cid2_pc_k = cid2_pc_k / sum(cid2_pc_k)
#         self.EMD_k = self.calcEMD(self.cid2_pc_k, self.cid2_pc)

        if debugging == True:
            print('digested group ' + str(self.k))
        self.ready = True