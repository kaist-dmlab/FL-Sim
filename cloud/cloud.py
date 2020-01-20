import numpy as np

import fl_data

# LINK_SPEED = '10Mbps', SIM_DATA_SIZE = 40000 도 비슷한 결과라서 시뮬레이션 빨리 하기 위해 SIM_DATA_SIZE 를 낮춤
# 시뮬레이션은 SIM_DATA_SIZE 크기가 클 수록 패킷 수가 늘어나서 오래걸림
# 자세한 결과는 test-struct.ipynb 참조
LINK_SPEED = '1MBps'
    
class Cloud:
    
    def __init__(self, ft, D_byNid, numGroups):
        self.ft = ft
        self.D_byNid = D_byNid
        numTotalClasses = len(np.unique(np.concatenate([D_byNid[nid]['y'] for nid in range(len(D_byNid))])))
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
        
        self.groups = [ Group(k, ft, D_byNid, cid2_pc, nid2_cid2_pc_is, nid2_cid2_numClasses_is) for k in range(numGroups) ]
        self.ps_nid = 0 # 모든 노드와의 거리가 같으므로 처음 노드로 설정
        self.ready = False
    
    def __eq__(self, other):
        if self.groups != other.groups:
            raise Exception(str(self.groups), str(other.groups))
        if self.ps_nid != other.ps_nid:
            raise Exception(str(self.ps_nid), str(other.ps_nid))
        if self.ready != other.ready:
            raise Exception(str(self.ready), str(other.ready))
        return True
    
    def clone(self):
        c_cloned = Cloud(self.ft, self.D_byNid, len(self.groups))
        c_cloned.set_p_is(self.get_p_is())
        c_cloned.digest(self.z)
        return c_cloned
        
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
        return list(self.nid2_p_i.values()) # 순서가 중요하지 않을 때 list 반환
    
    def get_nid2_D_i(self): # 전체 노드 별 데이터 집합
        if self.ready == False: raise Exception
        return self.nid2_D_i # 순서가 중요할 때 dict 반환
    
    def get_D_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_D_i.values()) # 순서가 중요하지 않을 때 list 반환
    
    def get_d_global(self, edgeCombineEnabled, linkSpeed=LINK_SPEED):
        if self.ready == False: raise Exception
        return self.ft.simulate('d_global', [ [nid, self.ps_nid] for nid in self.N ], edgeCombineEnabled, linkSpeed)
    
    def get_d_group(self, edgeCombineEnabled, linkSpeed=LINK_SPEED):
        if self.ready == False: raise Exception
        commPairs = []
        for g in self.groups:
            commPairs += [ [nid, g.ps_nid] for nid in g.get_N_k() ]
        return self.ft.simulate('d_group', commPairs, edgeCombineEnabled, linkSpeed)
    
    def get_hpp_global(self, edgeCombineEnabled):
        if self.ready == False: raise Exception
        return self.ft.getSumOfHopsPerPacket([ [nid, self.ps_nid] for nid in self.N ], edgeCombineEnabled)
    
    def get_hpp_group(self, edgeCombineEnabled):
        if self.ready == False: raise Exception
        return max([ self.ft.getSumOfHopsPerPacket([ [nid, g.ps_nid] for nid in g.get_N_k() ], edgeCombineEnabled) for g in self.groups ])
    
    def get_delta(self, nid2_g_i__w):
        if self.ready == False: raise Exception
        delta_ks = [ g.get_delta_k(nid2_g_i__w) for g in self.groups ]
        delta = np.average(delta_ks, weights=self.get_p_ks())
        return delta
    
    def get_Delta(self, nid2_g_i__w, g__w):
        if self.ready == False: raise Exception
        Delta_is = [ np.linalg.norm(nid2_g_i__w[nid] - g__w) for nid in self.get_N() ]
        Delta = np.average(Delta_is, weights=self.get_p_is())
        return Delta
    
    def get_DELTA(self, nid2_g_i__w, g__w):
        if self.ready == False: raise Exception
        DELTA_ks = [ g.get_DELTA_k(nid2_g_i__w, g__w) for g in self.groups ]
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
        
    def digest(self, z, debugging=False):
        self.z = z
        # Node Grouping 업데이트
        nids_byGid = fl_data.to_nids_byGid(z)
        if nids_byGid == None:
            self.ready = False
            return False
        else:
            for k, N_k in enumerate(nids_byGid):
                self.groups[k].set_N_k(N_k)
                
            # Digest
            self.N = []
            self.D = 0
            self.p_ks = []
            self.nid2_p_i = {}
            self.nid2_D_i = {}
            for g in self.groups:
                g.digest(debugging)
                self.N += g.get_N_k()
                self.D += g.get_p_k()
                self.p_ks.append(g.get_p_k())
                self.nid2_p_i.update(g.get_nid2_p_k_i()) # p_is 로 변환
                self.nid2_D_i.update(g.get_nid2_D_k_i())
            self.ready = True
            return True

class Group:
    
    def __init__(self, k, ft, D_byNid, cid2_pc, nid2_cid2_pc_is, nid2_cid2_numClasses_is):
        self.k = k
        self.ft = ft
        self.D_byNid = D_byNid
        self.p_is = None
        self.cid2_pc = cid2_pc
        self.nid2_cid2_pc_is = nid2_cid2_pc_is
        self.nid2_cid2_numClasses_is = nid2_cid2_numClasses_is
        self.N_k = []
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
        return self.nid2_p_k_i # 순서가 중요할 때 dict 반환
    
    def get_p_k_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_p_k_i.values()) # 순서가 중요하지 않을 때 list 반환
    
    def get_nid2_D_k_i(self): # 그룹 노드 별 데이터 집합
        if self.ready == False: raise Exception
        return self.nid2_D_k_i # 순서가 중요할 때 dict 반환
    
    def get_D_k_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_D_k_i.values()) # 순서가 중요하지 않을 때 list 반환
    
    def get_delta_k(self, nid2_g_i__w):
        if self.ready == False: raise Exception
        g_k_is__w = [ nid2_g_i__w[nid] for nid in self.get_N_k() ]
        g_k__w = np.average(g_k_is__w, axis=0, weights=self.get_p_k_is())
        delta_k_is = [ np.linalg.norm(nid2_g_i__w[nid] - g_k__w) for nid in self.get_N_k() ]
        delta_k = np.average(delta_k_is, weights=self.get_p_k_is())
        return delta_k
    
    def get_DELTA_k(self, nid2_g_i__w, g__w):
        if self.ready == False: raise Exception
        g_k_is__w = [ nid2_g_i__w[nid] for nid in self.get_N_k() ]
        g_k__w = np.average(g_k_is__w, axis=0, weights=self.get_p_k_is())
        DELTA_k = np.linalg.norm(g_k__w - g__w)
        return DELTA_k
        
    def calcEMD(self, a, b):
        if len(a) != len(b): raise Exception(len(a), len(b))
        return sum([ abs(a[i] - b[i]) for i in range(len(a)) ])
    
    def get_emd_k(self):
        if self.ready == False: raise Exception
        numTotalClasses = len(self.cid2_pc)
        cid2_pc_k = np.zeros(numTotalClasses, dtype=np.int32)
        for nid in self.N_k:
            cid2_pc_k += self.nid2_cid2_numClasses_is[nid]
        cid2_pc_k = cid2_pc_k / sum(cid2_pc_k)
        emd_k_is = [ self.calcEMD(self.nid2_cid2_pc_is[nid], cid2_pc_k) for nid in self.get_N_k() ]
        emd_k = np.average(emd_k_is, weights=self.get_p_k_is())
        return emd_k
    
    def get_EMD_k(self):
        if self.ready == False: raise Exception
        cid2_pc_k = np.zeros(numTotalClasses, dtype=np.int32)
        for nid in self.N_k:
            cid2_pc_k += self.nid2_cid2_numClasses_is[nid]
        cid2_pc_k = cid2_pc_k / sum(cid2_pc_k)
        EMD_k = self.calcEMD(self.cid2_pc_k, self.cid2_pc)
        return EMD_k
    
    def digest(self, debugging):
        if self.ready == True: return # Group 에 변화가 없을 때 연산되는 것을 방지
        self.p_k = 0
        self.nid2_p_k_i = {}
        self.nid2_D_k_i = {}
        for nid in self.N_k:
            D_k_i = self.D_byNid[nid]
            if self.p_is == None: # Weight p 미설정 시, 데이터 개수로 가중치
                p_k_i = len(D_k_i['x'])
            else:
                p_k_i = self.p_is[nid]
            self.p_k += p_k_i
            self.nid2_p_k_i[nid] = p_k_i
            self.nid2_D_k_i[nid] = D_k_i
        if not(debugging):
            self.ps_nid = self.N_k[ np.argmin([ sum( self.ft.getDistance(nid1, nid2) for nid2 in self.N_k ) for nid1 in self.N_k ]) ]
        self.ready = True