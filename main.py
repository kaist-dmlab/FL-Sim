import importlib
import traceback

import random
import numpy as np

import fl_util

alg = None

def main():
    args = fl_util.parseArgs()
    
    # tf.set_random_seed 는 model.abc 에서 Graph 생성 후 수행
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    
    algPackagePath = 'algorithm.' + args.algName
    algModule = importlib.import_module(algPackagePath)
    Algorithm = getattr(algModule, 'Algorithm')
    global alg
    alg = Algorithm(args)
    alg.run()
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        traceback.print_exc()
    alg.finalize()