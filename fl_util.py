import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algName',
                    help='algName',
                    type=str,
                    choices=['cgd', 'fedavg', 'hier_favg', 'ch_fedavg'],
                    required=True)
    parser.add_argument('--nodeType',
                    help='nodeType',
                    type=str,
                    default='one')
    parser.add_argument('--edgeType',
                    help='edgeType',
                    type=str,
                    default='one')
    parser.add_argument('--opaque1',
                    help='opaque1',
                    type=int,
                    default=1)
    parser.add_argument('--opaque2',
                    help='opaque2',
                    type=int,
                    default=1)
    return parser.parse_args()

def to_nids_byGid(z):
    gids = np.unique(z)
    for gid in range(len(gids)):
        if not(gid in gids): return None
    nids_byGid = [ [ nid for nid, gid in enumerate(z) if gid == gid_ ] for gid_ in gids ]
    return nids_byGid