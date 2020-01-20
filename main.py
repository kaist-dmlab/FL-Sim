import tensorflow as tf
import importlib

import random
import numpy as np

import fl_util

def main():
    args = fl_util.parseArgs()
    
    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)
    
    algPackagePath = 'algorithm.' + args.algName
    algModule = importlib.import_module(algPackagePath)
    Algorithm = getattr(algModule, 'Algorithm')
    alg = Algorithm(args)
    alg.run()
    
if __name__ == '__main__':
    main()