from wurlitzer import pipes
import subprocess

import networkx as nx
import ns.core
import ns.internet
import ns.applications
import ns.network

import numpy as np
import os

from abc import ABC, abstractmethod

import fl_const

LINK_DELAY = '1ms'
SIM_DATA_SIZE = 4000
PACKET_SIZE = 1000
PORT = 9
STOP_TIME = 10.0

class AbstractTopology(ABC):
    
    def __init__(self, modelSize, numNodes, numEdges):
        self.modelSize = modelSize
        self.numNodes = numNodes
        self.numEdges = numEdges
        
        self.g = nx.Graph()
        self.nid2eid = {}
        self.dist = {}
        
    def getDistance(self, nid1, nid2):
        return self.dist[nid1][nid2]
    
    def combineCommPairs(self, commPairs):
        # 현재 통신 Src 중 같은 Edge 에 속하며 같은 Dst 을 가지는 것에 대해 하나만 Src 로 Combine
        inEdgeCommPairs = []
        combinedCommPairs = {}
        for commPair in commPairs:
            srcNid = commPair[0]
            dstNid = commPair[1]
            srcEid = self.nid2eid[srcNid]
            dstEid = self.nid2eid[dstNid]
            if srcEid == dstEid:
                # Edge 내부 통신일 경우 combine 미적용
                inEdgeCommPairs.append((srcNid, dstNid))
            else:
                # Edge 간 통신일 경우 combine 적용
                if not((srcEid, dstNid) in combinedCommPairs):
                    combinedCommPairs[(srcEid, dstNid)] = (srcNid, dstNid)
        commPairs = inEdgeCommPairs + list(combinedCommPairs.values())
        return commPairs
    
    def getSumOfHopsPerPacket(self, commPairs, edgeCombineEnabled):
        if edgeCombineEnabled:
            commPairs = self.combineCommPairs(commPairs)
        
        numPackets = self.modelSize / PACKET_SIZE
        commPairs = np.array(commPairs)
        sumHPPs = 0
        for commPair in commPairs:
            srcNid = commPair[0]
            dstNid = commPair[1]
            dist = self.getDistance(srcNid, dstNid)
            sumHPPs += dist / numPackets
        return sumHPPs
    
    @abstractmethod
    def createSimNetwork(self, linkSpeed, linkDelay):
        pass
    
    def getDelay(self, commPairs, edgeCombineEnabled, linkSpeed):
        if edgeCombineEnabled:
            commPairs = self.combineCommPairs(commPairs)
        
        # remove pcap files in advance
        for f in os.listdir(fl_const.PCAP_DIR_NAME):
            os.remove(os.path.join(fl_const.PCAP_DIR_NAME, f))
        
        (p2p, nodes, xips) = self.createSimNetwork(linkSpeed, LINK_DELAY)
        
        # Populate routing table
        ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()
        
        totalBytesSent = 0
        commPairs = np.array(commPairs)
        
        for commPair in commPairs:
            srcNid = commPair[0]
            dstNid = commPair[1]
            if srcNid == dstNid: continue # 통신이 Loopback 인 경우 통과
            sourceHelper = ns.applications.BulkSendHelper('ns3::TcpSocketFactory', ns.network.InetSocketAddress(xips[dstNid], PORT))
            sourceHelper.SetAttribute('MaxBytes', ns.core.UintegerValue(SIM_DATA_SIZE))
            sourceApps = sourceHelper.Install(nodes.Get(srcNid))
            sourceApps.Start(ns.core.Seconds(0.0))
            sourceApps.Stop(ns.core.Seconds(STOP_TIME))
            totalBytesSent += SIM_DATA_SIZE
        if totalBytesSent == 0: # 아무것도 보낸 것이 없을 경우 시뮬레이션 Destroy 후 종료
            ns.core.Simulator.Destroy()
            return 0
        
        sinkAppsList = []
        for dstNid in np.unique(commPairs[:,1]):
            sinkHelper = ns.applications.PacketSinkHelper('ns3::TcpSocketFactory', ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), PORT))
            sinkApps = sinkHelper.Install(nodes.Get(dstNid))
            sinkApps.Start(ns.core.Seconds(0.0))
            sinkApps.Stop(ns.core.Seconds(STOP_TIME))
            sinkAppsList.append(sinkApps)
            
        ascii = ns.network.AsciiTraceHelper()
        p2p.EnablePcap(os.path.join(fl_const.PCAP_DIR_NAME, 'topology'), nodes, False)
        
#         print('Total Bytes Sent :', totalBytesSent)
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
            # 10: Network Simulation 을 빨리 하기 위해 1MBps/1000크기(10MBps/10000크기와 유사한 결과)로 했으므로 데이터 크기를 10 더 나눠줌
            return d / (SIM_DATA_SIZE * 10) * self.modelSize
#         print('Total Bytes Received :', sum( ns.applications.PacketSink(sinkApps.Get(0)).GetTotalRx() for sinkApps in sinkAppsList ))
        
        c_log_file = open(os.path.join(fl_const.LOG_DIR_NAME, fl_const.C_LOG_FILE_NAME), 'a')
        with pipes(stdout=c_log_file, stderr=c_log_file):
            maxPcapTime = max( getPcapTime(os.path.join(fl_const.PCAP_DIR_NAME, fileName)) for fileName in os.listdir(fl_const.PCAP_DIR_NAME) )
        c_log_file.close()
        if maxPcapTime == -1: raise Exception()
        maxPcapSec = toSec(maxPcapTime)
        return maxPcapSec