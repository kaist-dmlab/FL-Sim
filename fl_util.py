import argparse
import numpy as np

import pickle
import json

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName',
                    help='modelName',
                    type=str,
                    choices=['sr', '1nn', '2nn', 'cnn'], # svm, resnet 제외
                    required=True)
    parser.add_argument('--dataName',
                    help='dataName',
                    type=str,
                    choices=['mnist-o', 'mnist-f', 'cifar10', 'femnist', 'celeba'],
                    required=True)
    parser.add_argument('--algName',
                    help='algName',
                    type=str,
                    choices=['cgd', 'fedavg', 'hier-favg', 'ch-fedavg'],
                    required=True)
    parser.add_argument('--numNodeClasses',
                    help='numNodeClasses',
                    type=int,
                    required=True)
    parser.add_argument('--numEdgeClasses',
                    help='numEdgeClasses',
                    type=int,
                    required=True)
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
                    default=False)
    parser.add_argument('--maxEpoch',
                    help='maxEpoch',
                    type=int,
                    default=1000)
    parser.add_argument('--maxTime',
                    help='maxTime',
                    type=int,
                    default=-1)
    parser.add_argument('--lrInitial',
                    help='lrInitial',
                    type=float,
                    default=0.1)
    parser.add_argument('--lrDecayRate',
                    help='lrDecayRate',
                    type=int,
                    default=0.99)
    parser.add_argument('--numTestSamples',
                    help='numTestSamples',
                    type=int,
                    default=10000)
    parser.add_argument('--batchSize',
                    help='batchSize',
                    type=int,
                    default=128)
    parser.add_argument('--isValidation',
                    help='isValidation',
                    type=bool,
                    default=False)
    args = parser.parse_args()
    
    if args.modelName == 'svm':
        args.maxTime = 25
        args.lrInitial = 0.01
    elif args.modelName == 'sr':
        args.maxTime = 50
        if args.dataName == 'cifar10': # sr, cifar10 의 경우는 데이터에 비해 모델이 너무 단순해서 실험에서 제외
            raise Exception(args.modelName, args.dataName)
    elif args.modelName == '1nn':
        args.sgdEnabled = True
        args.maxTime = 250
    elif args.modelName == '2nn':
        args.sgdEnabled = True
        args.maxTime = 500
    elif args.modelName == 'cnn':
        args.sgdEnabled = True
        if args.dataName == 'mnist-o' or args.dataName == 'mnist-f' or args.dataName == 'femnist' or args.dataName == 'celeba':
            args.maxTime = 1000
        else:
            raise Exception(args.modelName, args.dataName)
    elif args.modelName == 'resnet':
        args.sgdEnabled = True
        args.maxTime = 10000
        args.batchSize = 256
        if args.dataName != 'cifar10': # 'resnet' 은 cifar10 만 지원
            raise Exception(args.modelName, args.dataName)
    else:
        raise Exception(args.modelName)
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