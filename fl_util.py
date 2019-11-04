import argparse
import numpy as np

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName',
                    help='modelName',
                    type=str,
                    choices=['svm', 'sr', '2nn', 'cnn', 'cnn_cifar10'],
                    required=True)
    parser.add_argument('--dataName',
                    help='dataName',
                    type=str,
                    choices=['mnist-o', 'mnist-f', 'cifar10'],
                    required=True)
    parser.add_argument('--algName',
                    help='algName',
                    type=str,
                    choices=['cgd', 'fedavg', 'hier_favg', 'ch_fedavg'],
                    required=True)
    parser.add_argument('--nodeType',
                    help='nodeType',
                    type=str,
                    choices=['o', 'f', 'h', 'a'],
                    required=True)
    parser.add_argument('--edgeType',
                    help='edgeType',
                    type=str,
                    choices=['o', 'f', 'h', 'a'],
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
    parser.add_argument('--sgdEnabled',
                    help='sgdEnabled',
                    type=bool,
                    default=False)
    parser.add_argument('--flatten',
                    help='flatten',
                    type=bool,
                    default=True)
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
                    type=int,
                    default=-1)
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
    args = parser.parse_args()
    
    if args.modelName == 'svm':
        args.maxTime = 25
        args.lrInitial = 0.01
    elif args.modelName == 'sr':
        args.maxTime = 50
        if args.dataName == 'mnist-o' or args.dataName == 'mnist-f':
            args.lrInitial = 0.1
    #     if args.dataName == 'cifar10': # 'sr', 'cifar10' 의 경우는 데이터에 비해 모델이 너무 단순해서 실험에서 제외
    #         args.lrInitial = 0.01
        else:
            raise Exception(args.modelName, args.dataName)
    elif args.modelName == '2nn':
        args.sgdEnabled = True
        args.maxTime = 500
        args.lrInitial = 0.1
    elif args.modelName == 'cnn':
        args.sgdEnabled = True
        args.flatten = False
        if args.dataName == 'mnist-o' or args.dataName == 'mnist-f':
            args.maxTime = 1000
            args.lrInitial = 0.1
        else:
            raise Exception(args.modelName, args.dataName)
    elif args.modelName == 'cnn_cifar10':
        args.sgdEnabled = True
        args.flatten = False
        if args.dataName == 'cifar10':
            args.maxTime = 10000
            args.lrInitial = 0.01
        else:
            raise Exception(args.modelName, args.dataName)
    else:
        raise Exception(args.modelName)
    return args

def to_nids_byGid(z):
    gids = np.unique(z)
    for gid in range(len(gids)):
        if not(gid in gids): return None
    nids_byGid = [ [ nid for nid, gid in enumerate(z) if gid == gid_ ] for gid_ in gids ]
    return nids_byGid