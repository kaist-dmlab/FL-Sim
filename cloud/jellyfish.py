import networkx as nx
import ns.core
import ns.point_to_point
import ns.internet
import ns.network

import random

from cloud.topology import AbstractTopology

DEGREE = 1

class Topology(AbstractTopology):
    
    def __init__(self, numNodes, numEdges):
        super().__init__(numNodes, numEdges)
        
        # 2012 Jellyfish: Networking Data Centers Randomly
        # N(k-r)
        # N = number of switches(numEdges)
        # k = number of total switch ports
        # r = network degree
        # k-r = number of nodes per switch(numNodesPerEdge)
        self.numNodesPerEdge = int(numNodes / numEdges)
        
        self.eidPairSet = set()
        for eid1 in range(numEdges):
            for _ in range(DEGREE):
                eid2 = random.randint(0, numEdges-1)
                eidPair = tuple(sorted([eid1, eid2]))
                while eid1 == eid2 or eidPair in self.eidPairSet:
                    eid2 = random.randint(0, numEdges-1)
                    eidPair = tuple(sorted([eid1, eid2]))
                self.g.add_edge('e' + str(eidPair[0]), 'e' + str(eidPair[1]))
                self.eidPairSet.add(eidPair)
        
        for eid in range(numEdges):
            for nid_ in range(self.numNodesPerEdge):
                nid = eid * self.numNodesPerEdge + nid_
                self.g.add_edge('e' + str(eid), nid)
                self.nid2eid[nid] = eid # For combineCommPairs()
        
        # Initialize hop distance
        for nid1 in range(self.numNodes):
            for nid2 in range(self.numNodes):
                if not(nid1 in self.dist):
                    self.dist[nid1] = {}
                self.dist[nid1][nid2] = len(nx.shortest_path(self.g, nid1, nid2)) - 1
                
    def createSimNetwork(self, linkSpeedStr, linkDelay):
        nodes = ns.network.NodeContainer()
        nodes.Create(self.numNodes)
        edges = ns.network.NodeContainer()
        edges.Create(self.numEdges)
        
        stack = ns.internet.InternetStackHelper()
        stack.Install(nodes)
        stack.Install(edges)
        
        def addrEdgeEdge(eid1, eid2):
            return "%d.%d.0.0" % (eid1 + 101, eid2 + 1)
        
        def addrEdgeNode(eid, nid):
            return "%d.%d.0.0" % (eid + 1, nid + 1)
        
        p2p = ns.point_to_point.PointToPointHelper()
        p2p.SetDeviceAttribute('DataRate', ns.core.StringValue(linkSpeedStr))
        p2p.SetChannelAttribute('Delay', ns.core.StringValue(linkDelay))
        
        for (eid1, eid2) in self.eidPairSet:
            ndc = p2p.Install(edges.Get(eid1), edges.Get(eid2))
            address = ns.internet.Ipv4AddressHelper()
            address.SetBase(ns.network.Ipv4Address(addrEdgeEdge(eid1, eid2)), ns.network.Ipv4Mask('255.255.255.0'))
            address.Assign(ndc)
            
        xips = []
        for eid in range(self.numEdges):
            for nid_ in range(self.numNodesPerEdge):
                nid = eid * self.numNodesPerEdge + nid_
                ndc = p2p.Install(edges.Get(eid), nodes.Get(nid))
                address = ns.internet.Ipv4AddressHelper()
                address.SetBase(ns.network.Ipv4Address(addrEdgeNode(eid, nid)), ns.network.Ipv4Mask('255.255.255.0'))
                ips = address.Assign(ndc)
                xips.append(ips.GetAddress(1))
        return p2p, nodes, xips