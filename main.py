import importlib

import random
import numpy as np

import fl_util

def main():
    args = fl_util.parseArgs()

    # Set the random seed if provided (affects client sampling, and batching)
    # tf.set_random_seed 는 model.abc 에서 Graph 생성 후 수행
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    
    algPackagePath = 'algorithm.' + args.algName
    algModule = importlib.import_module(algPackagePath)
    Algorithm = getattr(algModule, 'Algorithm')
    alg = Algorithm(args)
    alg.run()
    
if __name__ == '__main__':
    main()