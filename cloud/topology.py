from wurlitzer import pipes
import os
import subprocess
import numpy as np
import re

import networkx as nx
import ns.core
import ns.internet
import ns.applications
import ns.network

from abc import ABC, abstractmethod
import fl_const

LINK_DELAY = '1ms'
PORT = 9
STOP_TIME = 10.0

class AbstractTopology(ABC):
    
    def __init__(self, numNodes, numEdges):
        self.numNodes = numNodes
        self.numEdges = numEdges
        
        self.g = nx.Graph()
        self.dist = {}
        
    def getDistance(self, nid1, nid2):
        return self.dist[nid1][nid2]
    
    @abstractmethod
    def getEid(self, nid):
        pass
    
    @abstractmethod
    def getNids(self, eid):
        pass
    
    @abstractmethod
    def checkIfInSameEdge(self, nid1, nid2):
        pass
    
    def combineCommPairs(self, commPairs):
        # 현재 통신 Src 중 같은 Edge 에 속하며 같은 Dst 을 가지는 것에 대해 하나만 Src 로 Combine
        inEdgeCommPairs = []
        combinedCommPairs = {}
        for commPair in commPairs:
            srcNid = commPair[0]
            dstNid = commPair[1]
            srcEid = self.nid2eid[srcNid]
            if self.checkIfInSameEdge(srcNid, dstNid):
                # Edge 내부 통신일 경우 combine 미적용
                inEdgeCommPairs.append((srcNid, dstNid))
            else:
                # Edge 간 통신일 경우 combine 적용
                if not((srcEid, dstNid) in combinedCommPairs):
                    combinedCommPairs[(srcEid, dstNid)] = (srcNid, dstNid)
        commPairs = inEdgeCommPairs + list(combinedCommPairs.values())
        return commPairs
    
    def getSumOfHopsPerPacket(self, commPairs, edgeCombineEnabled, dataSize=fl_const.DEFAULT_PACKET_SIZE):
        if edgeCombineEnabled:
            commPairs = self.combineCommPairs(commPairs)
        
        numPackets = dataSize / fl_const.DEFAULT_PACKET_SIZE
        commPairs = np.array(commPairs)
        sumHPPs = 0
        for commPair in commPairs:
            srcNid = commPair[0]
            dstNid = commPair[1]
            dist = self.getDistance(srcNid, dstNid)
            sumHPPs += dist / numPackets
        return sumHPPs
    
    @abstractmethod
    def createSimNetwork(self, linkSpeedStr, linkDelay):
        pass
        
    def getPcapTime(self, filePath):
        proc = subprocess.Popen(['tcpdump', '-nn', '-tt', '-r', filePath], stdout=subprocess.PIPE)
        lastLine = None
        for line in proc.stdout.readlines():
            if len(line) > 5:
                lastLine = line
        if lastLine == None:
            return -1
        else:
            return float(lastLine.split()[0])
    
    def profileDelay(self, packetSize=fl_const.DEFAULT_PACKET_SIZE, linkSpeed=fl_const.DEFAULT_LINK_SPEED):
        linkSpeedStr = str(int(linkSpeed)) + 'MBps'
        
        commPairs = []
        for nid1 in range(self.numNodes):
            for nid2 in range(self.numNodes):
                if nid1 == nid2: continue
                commPairs.append((nid1, nid2))
        
        # remove pcap files in advance
        for f in os.listdir(fl_const.PCAP_DIR_PATH):
            os.remove(os.path.join(fl_const.PCAP_DIR_PATH, f))
        
        (p2p, nodes, xips) = self.createSimNetwork(linkSpeedStr, LINK_DELAY)
        
        # Populate routing table
        ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()
        
        totalBytesSent = 0
        commPairs = np.array(commPairs)
        
        for commPair in commPairs:
            srcNid = commPair[0]
            dstNid = commPair[1]
            if srcNid == dstNid: continue # 통신이 Loopback 인 경우 통과
            sourceHelper = ns.applications.BulkSendHelper('ns3::TcpSocketFactory', ns.network.InetSocketAddress(xips[dstNid], PORT))
            sourceHelper.SetAttribute('MaxBytes', ns.core.UintegerValue(packetSize))
            sourceApps = sourceHelper.Install(nodes.Get(srcNid))
            sourceApps.Start(ns.core.Seconds(0.0))
            sourceApps.Stop(ns.core.Seconds(STOP_TIME))
            totalBytesSent += packetSize
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
        p2p.EnablePcap(os.path.join(fl_const.PCAP_DIR_PATH, 'topology'), nodes, False)
        
#         print('Total Bytes Sent :', totalBytesSent)
        ns.core.Simulator.Stop(ns.core.Seconds(STOP_TIME))
        ns.core.Simulator.Run()
        ns.core.Simulator.Destroy()
#         print('Total Bytes Received :', sum( ns.applications.PacketSink(sinkApps.Get(0)).GetTotalRx() for sinkApps in sinkAppsList ))
        
        nid2delay = {}
        c_log_file = open(os.path.join(fl_const.LOG_DIR_PATH, fl_const.C_LOG_FILE_NAME), 'a')
        with pipes(stdout=c_log_file, stderr=c_log_file):
            for fileName in os.listdir(fl_const.PCAP_DIR_PATH):
                nid = int(re.match("topology-(\d+)-1.pcap", fileName).group(1))
                pcapSec = self.getPcapTime(os.path.join(fl_const.PCAP_DIR_PATH, fileName))
                if pcapSec == -1: raise Exception
                nid2delay[nid] = pcapSec
        c_log_file.close()
        return nid2delay
    
    def getDelay(self, commPairs, edgeCombineEnabled, dataSize=fl_const.DEFAULT_PACKET_SIZE, linkSpeed=fl_const.DEFAULT_LINK_SPEED):
        linkSpeedStr = str(int(linkSpeed)) + 'MBps'
        
        if edgeCombineEnabled:
            commPairs = self.combineCommPairs(commPairs)
        
        # remove pcap files in advance
        for f in os.listdir(fl_const.PCAP_DIR_PATH):
            os.remove(os.path.join(fl_const.PCAP_DIR_PATH, f))
        
        (p2p, nodes, xips) = self.createSimNetwork(linkSpeedStr, LINK_DELAY)
        
        # Populate routing table
        ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()
        
        totalBytesSent = 0
        commPairs = np.array(commPairs)
        
        for commPair in commPairs:
            srcNid = commPair[0]
            dstNid = commPair[1]
            if srcNid == dstNid: continue # 통신이 Loopback 인 경우 통과
            sourceHelper = ns.applications.BulkSendHelper('ns3::TcpSocketFactory', ns.network.InetSocketAddress(xips[dstNid], PORT))
            sourceHelper.SetAttribute('MaxBytes', ns.core.UintegerValue(fl_const.DEFAULT_PACKET_SIZE))
            sourceApps = sourceHelper.Install(nodes.Get(srcNid))
            sourceApps.Start(ns.core.Seconds(0.0))
            sourceApps.Stop(ns.core.Seconds(STOP_TIME))
            totalBytesSent += fl_const.DEFAULT_PACKET_SIZE
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
        p2p.EnablePcap(os.path.join(fl_const.PCAP_DIR_PATH, 'topology'), nodes, False)
        
#         print('Total Bytes Sent :', totalBytesSent)
        ns.core.Simulator.Stop(ns.core.Seconds(STOP_TIME))
        ns.core.Simulator.Run()
        ns.core.Simulator.Destroy()
        
        def toSec(d):
            return d / fl_const.DEFAULT_PACKET_SIZE * dataSize
#         print('Total Bytes Received :', sum( ns.applications.PacketSink(sinkApps.Get(0)).GetTotalRx() for sinkApps in sinkAppsList ))
        
        c_log_file = open(os.path.join(fl_const.LOG_DIR_PATH, fl_const.C_LOG_FILE_NAME), 'a')
        with pipes(stdout=c_log_file, stderr=c_log_file):
            maxPcapTime = max( self.getPcapTime(os.path.join(fl_const.PCAP_DIR_PATH, fileName)) for fileName in os.listdir(fl_const.PCAP_DIR_PATH) )
        c_log_file.close()
        if maxPcapTime == -1: raise Exception
        maxPcapSec = toSec(maxPcapTime)
        return maxPcapSec