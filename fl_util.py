import argparse
import numpy as np

import pickle
import json

from leaf.models.utils.model_utils import read_dir

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName',
                    help='modelName',
                    type=str,
                    choices=['sr', '1nn', '2nn', 'cnn-mnist', 'cnn-cifar10', 'cnn-femnist', 'cnn-celeba'], # svm, resnet 제외
                    required=True)
    parser.add_argument('--dataName',
                    help='dataName',
                    type=str,
                    choices=['mnist-o', 'mnist-f', 'femnist', 'celeba'], # cifar10 제외
                    required=True)
    parser.add_argument('--algName',
                    help='algName',
                    type=str,
                    choices=['cgd', 'fedavg', 'hier-favg', 'ch-fedavg', 'ch-fedavg-d'],
                    required=True)
    parser.add_argument('--nodeType',
                    help='nodeType',
                    type=str,
                    choices=['t', 'q', 'h', 'a'],
                    required=True)
    parser.add_argument('--edgeType',
                    help='edgeType',
                    type=str,
                    choices=['t', 'q', 'h', 'a'],
                    default='a')
    parser.add_argument('--opaque1',
                    help='opaque1',
                    type=float,
                    default=-1)
    parser.add_argument('--opaque2',
                    help='opaque2',
                    type=float,
                    default=-1)
    parser.add_argument('--numNodes',
                    help='numNodes',
                    type=int,
                    default=100)
    parser.add_argument('--numEdges',
                    help='numEdges',
                    type=int,
                    default=10)
    parser.add_argument('--numGroups',
                    help='numGroups',
                    type=int,
                    default=10)
    parser.add_argument('--sgdEnabled',
                    help='sgdEnabled',
                    type=bool,
                    default=True)
    parser.add_argument('--maxEpoch',
                    help='maxEpoch',
                    type=int,
                    default=1000)
    parser.add_argument('--maxTime',
                    help='maxTime',
                    type=int,
                    default=1000)
    # https://deviceatlas.com/blog/most-used-smartphone-gpu
    # https://en.wikipedia.org/wiki/PowerVR
    parser.add_argument('--procSpeeds',
                    help='procSpeeds',
                    type=int,
                    nargs='+',
                    default=[250]) # GFLOPS unit
    parser.add_argument('--linkSpeeds',
                    help='linkSpeeds',
                    type=int,
                    nargs='+',
                    default=[10]) # MBps unit
    parser.add_argument('--lrInitial',
                    help='lrInitial',
                    type=float,
                    default=0.1)
    parser.add_argument('--lrDecayRate',
                    help='lrDecayRate',
                    type=int,
                    default=0.99)
    parser.add_argument('--batchSize',
                    help='batchSize',
                    type=int,
                    default=128)
    parser.add_argument('--seed',
                    help='seed',
                    type=int,
                    default=0)
    parser.add_argument('--isValidation',
                    help='isValidation',
                    type=bool,
                    default=True)
    parser.add_argument('--topologyName',
                    help='topologyName',
                    type=str,
                    choices=['fattree', 'jellyfish'],
                    default='jellyfish')
    args = parser.parse_args()
    
    if args.modelName == 'sr':
        if args.dataName == 'cifar10': raise Exception(args.modelName, args.dataName)
        args.maxTime = 100
        args.sgdEnabled = False
    elif args.modelName == '1nn':
        if args.dataName == 'cifar10': raise Exception(args.modelName, args.dataName)
        args.maxTime = 1000
        args.sgdEnabled = False
    elif args.modelName == '2nn':
        if args.dataName == 'cifar10': raise Exception(args.modelName, args.dataName)
        args.maxTime = 1000
        args.sgdEnabled = False
    elif args.modelName == 'cnn-mnist':
        if not(args.dataName == 'mnist-o'
               or args.dataName == 'mnist-f'
               or args.dataName == 'femnist'): raise Exception(args.modelName, args.dataName)
        args.maxTime = 2000
    elif args.modelName == 'cnn-cifar10':
        if args.dataName != 'cifar10': raise Exception(args.modelName, args.dataName)
        args.maxTime = 2000
    elif args.modelName == 'cnn-femnist':
        if args.dataName != 'femnist': raise Exception(args.modelName, args.dataName)
        args.maxTime = 2000
    elif args.modelName == 'cnn-celeba':
        if args.dataName != 'celeba': raise Exception(args.modelName, args.dataName)
        args.maxTime = 2000
    else:
        raise Exception(args.modelName)
        
    if args.dataName == 'cifar10':
        args.lrInitial = 0.01
    elif args.dataName == 'femnist':
        args.lrInitial = 0.06 # LEAF Paper
    elif args.dataName == 'celeba':
        args.lrInitial = 0.001 # LEAF Paper
    return args

def serialize(filePath, obj):
    with open(filePath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def deserialize(filePath):
    with open(filePath, 'rb') as handle:
        return pickle.load(handle)
    
def dumpJson(filePath, jsonObj):
    with open(filePath, "w") as handle:
        json.dump(jsonObj, handle)

def loadJson(filePath):
    with open(filePath, 'r') as handle:
        return json.load(handle)
    
def readJsonDir(dirPath):
    return read_dir(dirPath)