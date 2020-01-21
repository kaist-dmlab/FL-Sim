import networkx as nx
import ns.core
import ns.point_to_point
import ns.internet
import ns.network

from cloud.topology import AbstractTopology

class Topology(AbstractTopology):
    
    def __init__(self, modelSize, numNodes, numEdges):
        super().__init__(modelSize, numNodes, numEdges)
        
        # 실험 상의 Edge 의 개수(numEdges)를 입력받지만, 실험 편의성을 높이기 위해
        # Fat Tree 내에서의 Pod 의 개수(numPods)를 실험 상의 Edge 의 개수(numEdges)로 간주한다.
        # Fat Tree 내에서의 Edge 의 개수(numFtEdges)는 정의에 의해서 결정된다.
        self.numPods = numEdges #self.K_half * 2
        
        self.K_half = int(self.numPods / 2)
        self.numCores = self.K_half * self.K_half
        #self.numCores = 1
        self.numAggrsPerPod = self.K_half
        self.numAggrs = self.numAggrsPerPod * self.numPods
        self.numFtEdgesPerPod = self.K_half
        self.numFtEdges = self.numFtEdgesPerPod * self.numPods # Fat Tree 내에서의 Edge 의 의미는 실험상의 Edge 의 의미와 다르므로 이름을 바꿔줌
#         self.numNodes = numNodes #self.numNodesPerPod * self.numPods
        self.numNodesPerEdge = int(numNodes / self.numFtEdges) #self.K_half
        self.numNodesPerPod = int(numNodes / self.numPods) #self.numNodesPerEdge * self.numFtEdgesPerPod
        #print(self.numPods, self.numCores, self.numAggrs, self.numFtEdges, self.numNodes)
        if not(self.numNodesPerPod % self.numNodesPerEdge == 0): raise Exception(str(self.numNodesPerPod) + ' ' + str(self.numNodesPerEdge))
        
        for pid in range(self.numPods):
            for cid in range(self.numCores):
                aid = pid * self.numAggrsPerPod + int(cid / self.K_half)
                #aid = pid * self.numAggrsPerPod + cid
                self.g.add_edge('c' + str(cid), 'a' + str(aid))
                
        for pid in range(self.numPods):
            for aid_ in range(self.numAggrsPerPod):
                for eid_ in range(self.numFtEdgesPerPod):
                    aid = pid * self.numAggrsPerPod + aid_
                    eid = pid * self.numFtEdgesPerPod + eid_
                    self.g.add_edge('a' + str(aid), 'e' + str(eid))
                    
        for pid in range(self.numPods):
            for eid_ in range(self.numFtEdgesPerPod):
                for nid_ in range(self.numNodesPerEdge):
                    eid = pid * self.numFtEdgesPerPod + eid_
                    nid = pid * self.numNodesPerPod + eid_ * self.numNodesPerEdge + nid_
                    self.g.add_edge('e' + str(eid), nid)
                    self.nid2eid[nid] = eid # For combineCommPairs()
                    
        # Initialize hop distance
        for nid1 in range(self.numNodes):
            for nid2 in range(self.numNodes):
                if not(nid1 in self.dist):
                    self.dist[nid1] = {}
                self.dist[nid1][nid2] = len(nx.shortest_path(self.g, nid1, nid2)) - 1
        
    def createSimNetwork(self, linkSpeed, linkDelay):
        cores = ns.network.NodeContainer()
        cores.Create(self.numCores)
        aggrs = ns.network.NodeContainer()
        aggrs.Create(self.numAggrs)
        edges = ns.network.NodeContainer()
        edges.Create(self.numFtEdges)
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
        p2p.SetDeviceAttribute('DataRate', ns.core.StringValue(linkSpeed))
        p2p.SetChannelAttribute('Delay', ns.core.StringValue(linkDelay))
        
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
                for eid_ in range(self.numFtEdgesPerPod):
                    aid = pid * self.numAggrsPerPod + aid_
                    eid = pid * self.numFtEdgesPerPod + eid_
                    ndc = p2p.Install(aggrs.Get(aid), edges.Get(eid))
                    
                    address = ns.internet.Ipv4AddressHelper()
                    address.SetBase(ns.network.Ipv4Address(addrAggrEdge(pid, aid, eid)), ns.network.Ipv4Mask('255.255.255.0'))
                    address.Assign(ndc)
                    
        xips = []
        cntNodes = 0
        for pid in range(self.numPods):
            for eid_ in range(self.numFtEdgesPerPod):
                for nid_ in range(self.numNodesPerEdge):
                    if cntNodes < self.numNodes:
                        cntNodes += 1
                    else:
                        break
                    eid = pid * self.numFtEdgesPerPod + eid_
                    nid = pid * self.numNodesPerPod + eid_ * self.numNodesPerEdge + nid_
                    ndc = p2p.Install(edges.Get(eid), nodes.Get(nid))
                    
                    address = ns.internet.Ipv4AddressHelper()
                    address.SetBase(ns.network.Ipv4Address(addrEdgeNode(pid, eid, nid)), ns.network.Ipv4Mask('255.255.255.0'))
                    ips = address.Assign(ndc)
                    xips.append(ips.GetAddress(1))
        return p2p, nodes, xips