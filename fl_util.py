import argparse
import os
import numpy as np
import random
import pickle
import json

from leaf.models.utils.model_utils import read_dir
import fl_const

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName',
                    help='modelName',
                    type=str,
                    choices=['sr', '2nn', 'cnn-mnist', 'cnn-celeba'], # 1nn, svm, cnn-femnist, resnet 제외
                    required=True)
    parser.add_argument('--dataName',
                    help='dataName',
                    type=str,
                    choices=['mnist-o', 'mnist-f', 'femnist', 'celeba'], # cifar10 제외
                    required=True)
    parser.add_argument('--algName',
                    help='algName',
                    type=str,
                    choices=['cgd', 'fedavg', 'hier-favg', 'fedavg-i', 'fedavg-c', 'fedavg-ic'], # fedavg-ic2, fedavg-ss 제외
                    required=True)
    parser.add_argument('--nodeType',
                    help='nodeType',
                    type=str,
                    choices=['t', 'q', 'h'], # a 제외
                    default='t')
    parser.add_argument('--edgeType',
                    help='edgeType',
                    type=str,
                    choices=['t', 'q', 'h'], # a 제외
                    default='h')
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
                    default=False)
    parser.add_argument('--topologyName',
                    help='topologyName',
                    type=str,
                    choices=['fattree', 'jellyfish'],
                    default='fattree')
    args = parser.parse_args()
    
    if args.modelName == 'sr':
        args.sgdEnabled = False
        args.maxTime = 100
    elif args.modelName == '2nn':
        args.sgdEnabled = False
        if args.dataName == 'mnist-o' or args.dataName == 'mnist-f':
            args.maxTime = 3000
        else:
            raise Exception(args.modelName, args.dataName)
    elif args.modelName == 'cnn-mnist':
        if args.dataName == 'mnist-o':
            args.maxTime = 6000
        elif args.dataName == 'mnist-f':
            args.maxTime = 8000
        elif args.dataName == 'femnist':
            args.maxTime = 6000 # TODO
        else:
            raise Exception(args.modelName, args.dataName)
    elif args.modelName == 'cnn-celeba':
        if args.dataName == 'celeba':
            args.maxTime = 2000
            args.lrInitial = 0.001 # LEAF Paper
        else:
            raise Exception(args.modelName, args.dataName)
    else:
        raise Exception(args.modelName)
    return args

def initialize(seed=0):
    # Initialize C log file
    open(os.path.join(fl_const.LOG_DIR_PATH, fl_const.C_LOG_FILE_NAME), 'w')
    
    # tf.set_random_seed 는 model.abc 에서 Graph 생성 후 수행
    random.seed(1 + seed)
    np.random.seed(12 + seed)

def serialize(filePath, obj):
    with open(filePath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def deserialize(filePath):
    with open(filePath, 'rb') as handle:
        return pickle.load(handle)
    
def dumpJson(filePath, jsonObj):
    with open(filePath, "w") as handle:
        json.dump(jsonObj, handle, indent=2, sort_keys=True)

def loadJson(filePath):
    with open(filePath, 'r') as handle:
        return json.load(handle)
    
def readJsonDir(dirPath):
    return read_dir(dirPath)