import tensorflow as tf
import importlib

import fl_util

def main():
    args = fl_util.parseArgs()
    
    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    algPackagePath = 'algorithm.' + args.algName
    algModule = importlib.import_module(algPackagePath)
    Algorithm = getattr(algModule, 'Algorithm')
    alg = Algorithm(args)
    alg.run()
    
if __name__ == '__main__':
    main()