import networkx as nx
import numpy as np
import ns.core
import ns.point_to_point
import ns.internet
import ns.applications
import ns.network
import subprocess
import os

import fl_util

# LINKSPEED = '10Mbps', SIM_SIM_DATA_SIZE = 40000 도 비슷한 결과라서 시뮬레이션 빨리 하기 위해 SIM_DATA_SIZE 를 낮춤
# 시뮬레이션은 SIM_DATA_SIZE 크기가 클 수록 패킷 수가 늘어나서 오래걸림
# 자세한 결과는 test-struct.ipynb 참조
LINKSPEED = '1MBps'
LINKDELAY = '1ms'
SIM_DATA_SIZE = 4000
PORT = 9
STOP_TIME = 10.0

class FatTree:
    # numEdges 를 입력받으면 K/2 = int(sqrt(numEdges / 2)) 로써, 소수가 발생하기 때문에,
    # 실험 편의성을 높이기 위해 Pod 의 개수로 대체한다.
    def __init__(self, numNodes, numPods):
        K_half = int(numPods / 2)
        self.K_half = K_half
        self.numPods = numPods #self.K_half * 2
        self.numCores = self.K_half * self.K_half
        #self.numCores = 1
        self.numAggrsPerPod = self.K_half
        self.numAggrs = self.numAggrsPerPod * self.numPods
        self.numEdgesPerPod = self.K_half
        self.numEdges = self.numEdgesPerPod * self.numPods
        self.numNodes = numNodes #self.numNodesPerPod * self.numPods
        self.numNodesPerEdge = int(numNodes / self.numEdges) #self.K_half
        self.numNodesPerPod = int(numNodes / numPods) #self.numNodesPerEdge * self.numEdgesPerPod
        #print(self.numPods, self.numCores, self.numAggrs, self.numEdges, self.numNodes)
        if not(self.numNodesPerPod % self.numNodesPerEdge == 0): raise Exception(str(self.numNodesPerPod) + ' ' + str(self.numNodesPerEdge))
        
        self.g = nx.Graph()
        for pid in range(self.numPods):
            for cid in range(self.numCores):
                aid = pid * self.numAggrsPerPod + int(cid / self.K_half)
                #aid = pid * self.numAggrsPerPod + cid
                self.g.add_edge('c' + str(cid), 'a' + str(aid))
                
        for pid in range(self.numPods):
            for aid_ in range(self.numAggrsPerPod):
                for eid_ in range(self.numEdgesPerPod):
                    aid = pid * self.numAggrsPerPod + aid_
                    eid = pid * self.numEdgesPerPod + eid_
                    self.g.add_edge('a' + str(aid), 'e' + str(eid))
                    
        self.nid2eid = {}
        for pid in range(self.numPods):
            for eid_ in range(self.numEdgesPerPod):
                for nid_ in range(self.numNodesPerEdge):
                    eid = pid * self.numEdgesPerPod + eid_
                    nid = pid * self.numNodesPerPod + eid_ * self.numNodesPerEdge + nid_
                    self.g.add_edge('e' + str(eid), nid)
                    self.nid2eid[nid] = eid
                    
        # Initialize hop distance
        self.dist = { nid1: { nid2: len(nx.shortest_path(self.g, nid1, nid2)) - 1 for nid2 in range(self.numNodes) } for nid1 in range(self.numNodes) }
                
    def simulate(self, subject, commPairs, edgeCombineEnabled, modelSize):
        if edgeCombineEnabled:
            # 현재 통신 Src 중 같은 Edge 에 속하며 같은 Dst 을 가지는 것에 대해 하나만 Src 로 Combine
            combinedCommPairs = {}
            for commPair in commPairs:
                srcNid = commPair[0]
                dstNid = commPair[1]
                srcEid = self.nid2eid[srcNid]
                if not((srcEid, dstNid) in combinedCommPairs):
                    combinedCommPairs[(srcEid, dstNid)] = [srcNid, dstNid]
            commPairs = list(combinedCommPairs.values())
            
        def removeFilesInDir(dirPath):
            for f in os.listdir(dirPath):
                os.remove('pcap/' + f)
        removeFilesInDir('pcap')
        
        cores = ns.network.NodeContainer()
        cores.Create(self.numCores)
        aggrs = ns.network.NodeContainer()
        aggrs.Create(self.numAggrs)
        edges = ns.network.NodeContainer()
        edges.Create(self.numEdges)
        nodes = ns.network.NodeContainer()
        nodes.Create(self.numNodes)
        
        stack = ns.internet.InternetStackHelper()
        stack.Install(cores)
        stack.Install(aggrs)
        stack.Install(edges)
        stack.Install(nodes)
        
        def addrCoreAggr(cid, pid, aid):
            return "%d.%d.%d.0" % (cid + 1, pid + 1, aid + 1)
        
        def addrAggrEdge(pid, aid, eid):
            return "%d.%d.%d.0" % (pid + 101, aid + 1, eid + 1)
        
        def addrEdgeNode(pid, eid, nid):
            return "%d.%d.%d.0" % (pid + 201, eid + 1, nid + 1)
        
        p2p = ns.point_to_point.PointToPointHelper()
        p2p.SetDeviceAttribute('DataRate', ns.core.StringValue(LINKSPEED))
        p2p.SetChannelAttribute('Delay', ns.core.StringValue(LINKDELAY))
        
        for pid in range(self.numPods):
            for cid in range(self.numCores):
                aid = pid * self.numAggrsPerPod + int(cid / self.K_half)
                #aid = pid * self.numAggrsPerPod + cid
                ndc = p2p.Install(cores.Get(cid), aggrs.Get(aid))
                
                address = ns.internet.Ipv4AddressHelper()
                address.SetBase(ns.network.Ipv4Address(addrCoreAggr(cid, pid, aid)), ns.network.Ipv4Mask('255.255.255.0'))
                address.Assign(ndc)
                
        for pid in range(self.numPods):
            for aid_ in range(self.numAggrsPerPod):
                for eid_ in range(self.numEdgesPerPod):
                    aid = pid * self.numAggrsPerPod + aid_
                    eid = pid * self.numEdgesPerPod + eid_
                    ndc = p2p.Install(aggrs.Get(aid), edges.Get(eid))
                    
                    address = ns.internet.Ipv4AddressHelper()
                    address.SetBase(ns.network.Ipv4Address(addrAggrEdge(pid, aid, eid)), ns.network.Ipv4Mask('255.255.255.0'))
                    address.Assign(ndc)
                    
        self.xips = []
        cntNodes = 0
        for pid in range(self.numPods):
            for eid_ in range(self.numEdgesPerPod):
                for nid_ in range(self.numNodesPerEdge):
                    if cntNodes < self.numNodes:
                        cntNodes += 1
                    else:
                        break
                    eid = pid * self.numEdgesPerPod + eid_
                    nid = pid * self.numNodesPerPod + eid_ * self.numNodesPerEdge + nid_
                    ndc = p2p.Install(edges.Get(eid), nodes.Get(nid))
                    
                    address = ns.internet.Ipv4AddressHelper()
                    address.SetBase(ns.network.Ipv4Address(addrEdgeNode(pid, eid, nid)), ns.network.Ipv4Mask('255.255.255.0'))
                    ips = address.Assign(ndc)
                    self.xips.append(ips.GetAddress(1))
                    
        # Populate routing table
        ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()
        
        totalBytesSent = 0
        commPairs = np.array(commPairs)
        for commPair in commPairs:
            srcNid = commPair[0]
            dstNid = commPair[1]
            sourceHelper = ns.applications.BulkSendHelper('ns3::TcpSocketFactory', ns.network.InetSocketAddress(self.xips[dstNid], PORT))
            sourceHelper.SetAttribute('MaxBytes', ns.core.UintegerValue(SIM_DATA_SIZE))
            sourceApps = sourceHelper.Install(nodes.Get(srcNid))
            sourceApps.Start(ns.core.Seconds(0.0))
            sourceApps.Stop(ns.core.Seconds(STOP_TIME))
            totalBytesSent += SIM_DATA_SIZE
            
        sinkAppsList = []
        for dstNid in np.unique(commPairs[:,1]):
            sinkHelper = ns.applications.PacketSinkHelper('ns3::TcpSocketFactory', ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), PORT))
            sinkApps = sinkHelper.Install(nodes.Get(dstNid))
            sinkApps.Start(ns.core.Seconds(0.0))
            sinkApps.Stop(ns.core.Seconds(STOP_TIME))
            sinkAppsList.append(sinkApps)
            
        ascii = ns.network.AsciiTraceHelper()
        p2p.EnablePcap('pcap/fattree', nodes, False)
        
        #print('Total Bytes Sent :', totalBytesSent)
        ns.core.Simulator.Stop(ns.core.Seconds(STOP_TIME))
        ns.core.Simulator.Run()
        ns.core.Simulator.Destroy()
        
        def getPcapTime(filePath):
            proc = subprocess.Popen(['tcpdump', '-nn', '-tt', '-r', filePath], stdout=subprocess.PIPE)
            lastLine = None
            for line in proc.stdout.readlines():
                if len(line) > 5:
                    lastLine = line
            if lastLine == None:
                return -1
            else:
                return float(lastLine.split()[0])
        def toSec(d):
            # 10: Network Simulation 을 빨리 하기 위해 1MBps/1000크기(10MBps/10000크기와 유사한 결과)로 했으며, 따라서 데이터 크기를 10 더 나눠줌
            # 4: Number of bytes for a float variable
            return d / (SIM_DATA_SIZE * 10) * modelSize * 4
        #print('Total Bytes Received :', sum( ns.applications.PacketSink(sinkApps.Get(0)).GetTotalRx() for sinkApps in sinkAppsList ))
        maxPcapTime = max( getPcapTime('pcap/' + fileName) for fileName in os.listdir('pcap') )
        if maxPcapTime == -1: raise Exception()
        maxPcapSec = toSec(maxPcapTime)
        print('Simulation Finished(%s) : %.3f s' % (subject, maxPcapSec))
        return maxPcapSec
    
    def getDistance(self, nid1, nid2):
        return self.dist[nid1][nid2]
    
class Cloud:
    def __init__(self, ft, data_byNid, numGroups, modelSize=-1):
        self.ft = ft
        self.data_byNid = data_byNid
        self.modelSize = modelSize
        self.groups = [ Group(k, ft, data_byNid, modelSize) for k in range(numGroups) ]
        self.ps_nid = 0 # 모든 노드와의 거리가 같으므로 처음 노드로 설정
        self.ready = False
        
    def get_N(self): # 전체 노드 집합
        if self.ready == False: raise Exception
        return self.N
    
    def get_D(self): # 전체 데이터 크기
        if self.ready == False: raise Exception
        return self.D
    
    def get_D_ks(self): # 그룹 별 데이터 크기 집합
        if self.ready == False: raise Exception
        return self.D_ks
    
    def get_nid2_D_i(self): # 전체 노드 별 데이터 크기 집합
        if self.ready == False: raise Exception
        return self.nid2_D_i # 순서가 중요할 때 dict 반환
    
    def get_D_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_D_i.values()) # 순서가 중요하지 않을 때 list 반환
    
    def get_nid2_Data_i(self): # 전체 노드 별 데이터 집합
        if self.ready == False: raise Exception
        return self.nid2_Data_i # 순서가 중요할 때 dict 반환
    
    def get_Data_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_Data_i.values()) # 순서가 중요하지 않을 때 list 반환
    
    def get_d_global(self, edgeCombineEnabled):
        if self.ready == False: raise Exception
        return self.ft.simulate('d_global', [ [nid, self.ps_nid] for nid in self.N ], edgeCombineEnabled, self.modelSize)
    
    def get_d_group(self, edgeCombineEnabled):
        if self.ready == False: raise Exception
        commPairs = []
        for g in self.groups:
            commPairs += [ [nid, g.ps_nid] for nid in g.get_N_k() ]
        return self.ft.simulate('d_group', commPairs, edgeCombineEnabled, self.modelSize)
    
    def get_Delta(self, nid2_g_i__w, g__w):
        if self.ready == False: raise Exception
        Delta_ks = [ g.get_Delta_k(nid2_g_i__w, g__w) for g in self.groups ]
        Delta = np.average(Delta_ks, weights=self.get_D_ks())
        return Delta
        
    def digest(self, z, debugging=False):
        self.z = z
        # Node Grouping 업데이트
        nids_byGid = fl_util.to_nids_byGid(z)
        if nids_byGid == None:
            self.ready = False
            return False
        else:
            for k, N_k in enumerate(nids_byGid):
                self.groups[k].set_N_k(N_k)
                
            # Digest
            self.N = []
            self.D = 0
            self.D_ks = []
            self.nid2_D_i = {}
            self.nid2_Data_i = {}
            for g in self.groups:
                g.digest(debugging)
                self.N += g.get_N_k()
                self.D += g.get_D_k()
                self.D_ks.append(g.get_D_k())
                self.nid2_D_i.update(g.get_nid2_D_k_i())
                self.nid2_Data_i.update(g.get_nid2_Data_k_i())
            self.ready = True
            return True
        
class Group:
    def __init__(self, k, ft, data_byNid, modelSize):
        self.k = k
        self.ft = ft
        self.data_byNid = data_byNid
        self.modelSize = modelSize
        self.N_k = []
        self.ready = False
        
    def set_N_k(self, N_k):
        if not(self.N_k == N_k):
            self.N_k = N_k
            self.ready = False
        
    def get_N_k(self): # 그룹 노드 집합
        if self.ready == False: raise Exception
        return self.N_k
    
    def get_D_k(self): # 그룹 데이터 크기
        if self.ready == False: raise Exception
        return self.D_k
    
    def get_nid2_D_k_i(self): # 그룹 노드 별 데이터 크기 집합
        if self.ready == False: raise Exception
        return self.nid2_D_k_i # 순서가 중요할 때 dict 반환
    
    def get_D_k_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_D_k_i.values()) # 순서가 중요하지 않을 때 list 반환
    
    def get_nid2_Data_k_i(self): # 그룹 노드 별 데이터 집합
        if self.ready == False: raise Exception
        return self.nid2_Data_k_i # 순서가 중요할 때 dict 반환
    
    def get_Data_k_is(self):
        if self.ready == False: raise Exception
        return list(self.nid2_Data_k_i.values()) # 순서가 중요하지 않을 때 list 반환
    
    def get_Delta_k(self, nid2_g_i__w, g__w):
        if self.ready == False: raise Exception
        g_k_is__w = [ nid2_g_i__w[nid] for nid in self.get_N_k() ]
        g_k__w = np.average(g_k_is__w, axis=0, weights=self.get_D_k_is())
        Delta_k = np.linalg.norm(g_k__w - g__w)
        return Delta_k
    
    def digest(self, debugging):
        if self.ready == True: return # Group 에 변화가 없을 때 연산되는 것을 방지
        self.D_k = 0 ;
        self.nid2_D_k_i = {}
        self.nid2_Data_k_i = {}
        for nid in self.N_k:
            data_k_i = self.data_byNid[nid]
            self.D_k += len(data_k_i['x'])
            self.nid2_D_k_i[nid] = len(data_k_i['x'])
            self.nid2_Data_k_i[nid] = data_k_i
            
        if not(debugging):
            self.ps_nid = self.N_k[ np.argmin([ sum( self.ft.getDistance(nid1, nid2) for nid2 in self.N_k ) for nid1 in self.N_k ]) ]
        self.ready = True